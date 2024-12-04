import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time 

# learning a simple 1d two-bump distribution with an energy model
# just a minimal example to show how ebms work

class SimpleEBM(nn.Module):
    """small net that learns the energy landscape for 1d data"""
    def __init__(self, input_dim=1, hidden_dims=[32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # raw energy (low = more likely to be real, high = fake)
        return self.net(x)

def generate_target_data(n_samples=1000, pattern='two_bumps'):
    """Generate different data patterns
    
    Args:
        n_samples: Number of samples to generate
        pattern: One of ['two_bumps', 'ring', 'spiral']
    """
    if pattern == 'two_bumps':
        n_per_mode = n_samples // 2
        samples1 = np.random.normal(-2, 0.5, n_per_mode)
        samples2 = np.random.normal(2, 0.5, n_per_mode)
        samples = np.concatenate([samples1, samples2])
        return torch.FloatTensor(samples).reshape(-1, 1)
    
    elif pattern == 'ring':
        # Generate points in a ring shape
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        r = np.random.normal(2, 0.2, n_samples)  # radius ~2 with noise
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        samples = np.stack([x, y], axis=1)
        return torch.FloatTensor(samples)
    
    elif pattern == 'spiral':
        # Generate points in a spiral shape
        theta = np.random.uniform(0, 4*np.pi, n_samples)
        r = theta / (4*np.pi) * 3  # radius increases with angle
        noise = np.random.normal(0, 0.1, n_samples)
        x = (r + noise) * np.cos(theta)
        y = (r + noise) * np.sin(theta)
        samples = np.stack([x, y], axis=1)
        return torch.FloatTensor(samples)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

def train_step(model, optimizer, real_samples):
    """one training step using contrastive divergence"""
    batch_size = len(real_samples)
    input_dim = real_samples.shape[1]
    
    # Sample noise from a mixture of uniform and data-centered distributions
    noise_range = 4 if input_dim == 1 else 6
    uniform_noise = torch.FloatTensor(batch_size, input_dim).uniform_(-noise_range, noise_range)
    
    # Add some noise samples near the real data
    data_centered_noise = real_samples + torch.randn_like(real_samples) * 0.5
    noise = torch.cat([uniform_noise, data_centered_noise])
    real_samples = torch.cat([real_samples, real_samples])  # Match noise size
    
    # get energies for real and noise samples
    real_energy = model(real_samples)
    noise_energy = model(noise)
    
    # Use softplus - it gives smoother gradients and better numerical stability than tanh
    # since it doesn't saturate as quickly at the extremes
    loss = torch.nn.functional.softplus(real_energy).mean() + torch.nn.functional.softplus(-noise_energy).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def visualize_model(model, real_data, title):
    """show what the model learned"""
    input_dim = real_data.shape[1]
    
    if input_dim == 1:
        x = torch.linspace(-6, 6, 1000).reshape(-1, 1)
        
        with torch.no_grad():
            energies = model(x)
        
        plt.figure(figsize=(12, 4))
        
        # energy landscape plot
        plt.subplot(131)
        plt.plot(x.numpy(), energies.numpy())
        plt.title('Energy Landscape')
        plt.xlabel('x')
        plt.ylabel('Energy')
        
        # real data distribution
        plt.subplot(132)
        plt.hist(real_data.numpy(), bins=50, density=True, alpha=0.5, label='Real Data')
        plt.title('Data Distribution')
        plt.xlabel('x')
        plt.ylabel('Density')
        
    else:  # 2D visualization
        plt.figure(figsize=(12, 4))
        
        # energy landscape plot
        plt.subplot(131)
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)
        points = torch.FloatTensor(np.stack([X.flatten(), Y.flatten()], axis=1))
        
        with torch.no_grad():
            energies = model(points).reshape(100, 100)
        
        plt.contourf(X, Y, energies.numpy(), levels=20)
        plt.colorbar(label='Energy')
        plt.title('Energy Landscape')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # real data distribution
        plt.subplot(132)
        plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.1, s=1)
        plt.title('Data Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
    
    # samples from model using softmax sampling
    plt.subplot(133)
    with torch.no_grad():
        # generate candidate points
        candidates = torch.FloatTensor(50000, input_dim).uniform_(-10, 10)
        energies = model(candidates)
        
        # convert energies to probabilities with temperature
        temp = 0.3
        probs = torch.softmax(-energies/temp, dim=0)
        
        # sample 5k points according to probabilities
        idx = torch.multinomial(probs.view(-1), 10000, replacement=True)
        samples = candidates[idx]
        
        if input_dim == 1:
            plt.hist(samples.numpy(), bins=50, density=True, alpha=0.5)
        else:
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1)
            plt.axis('equal')
        plt.title('Sampled from Model')
        plt.xlabel('x')
        plt.ylabel('y' if input_dim > 1 else 'Density')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def find_best_architecture(pattern, training_time=5):
    """search for the fastest learning architecture by measuring loss improvement over fixed time"""
    real_data = generate_target_data(10000, pattern=pattern)
    input_dim = real_data.shape[1]
    

    # early experiments showed that wider models didnt help, so trying with thin and many layered models 
    architectures = [
        [8, 8, 8], [12, 12, 12], [16, 16, 16],  # three layers
        [8, 8, 8, 8], [12, 12, 12, 12], [16, 16, 16, 16],  # four layers 
        [8, 8, 8, 8, 8], [12, 12, 12, 12, 12], [16, 16, 16, 16, 16],  # five layers
        [8, 8, 8, 8, 8, 8], [12, 12, 12, 12, 12, 12], [16, 16, 16, 16, 16, 16],  # six layers
        [8, 8, 8, 8, 8, 8, 8], [12, 12, 12, 12, 12, 12, 12], [16, 16, 16, 16, 16, 16, 16]  # seven layers
    ]
    best_arch = None
    best_loss_curve = []
    best_loss_improvement = float('-inf')
    
    for hidden_dims in architectures:
        print(f"\nTrying architecture with hidden dims: {hidden_dims}")
        
        model = SimpleEBM(input_dim=input_dim, hidden_dims=hidden_dims)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        start_time = time.time()
        
        while time.time() - start_time < training_time:
            loss = train_step(model, optimizer, real_data)
            losses.append(loss)
            
        # calc rate of improvement over time
        initial_loss = sum(losses[:10]) / 10  # avg first 10 losses
        final_loss = sum(losses[-10:]) / 10   # avg last 10 losses
        loss_improvement = initial_loss - final_loss
        
        print(f"Loss improvement: {loss_improvement:.4f}")
        
        if loss_improvement > best_loss_improvement:
            best_loss_improvement = loss_improvement
            best_arch = hidden_dims
            best_loss_curve = losses
    
    print(f"\nBest architecture found: {best_arch}")
    print(f"Best loss improvement: {best_loss_improvement:.4f}")
    return best_arch

if __name__ == "__main__":
    # Choose pattern: 'two_bumps', 'ring', or 'spiral'
    pattern = 'spiral'
    
    # Find best architecture first
    print("Searching for best architecture...")
    best_hidden_dims = find_best_architecture(pattern)
    
    # Train with best architecture
    print("\nTraining with best architecture...")
    real_data = generate_target_data(10000, pattern=pattern)
    input_dim = real_data.shape[1]
    model = SimpleEBM(input_dim=input_dim, hidden_dims=best_hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # training settings
    n_epochs = 10000
    vis_every = 1000  # visualize less frequently
    
    # train the model
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        loss = train_step(model, optimizer, real_data)
        pbar.set_description(f"Loss: {loss:.4f}")
        
        if (epoch + 1) % vis_every == 0:
            visualize_model(model, real_data, f'EBM Training Progress - Epoch {epoch+1}')
