import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# learning a simple 1d two-bump distribution with an energy model
# just a minimal example to show how ebms work

class SimpleEBM(nn.Module):
    """small net that learns the energy landscape for 1d data"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Initialize scaling factor to help control energy range
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # raw energy (low = more likely to be real, high = fake)
        energy = self.net(x)
        
        # tanh is CRITICAL, scale probs unnecessary 
        scaled_energy = torch.tanh(energy / self.scale) * self.scale
        return scaled_energy

def generate_target_data(n_samples=1000):
    """make samples from two gaussians"""
    n_per_mode = n_samples // 2
    samples1 = np.random.normal(-2, 0.5, n_per_mode)
    samples2 = np.random.normal(2, 0.5, n_per_mode)
    samples = np.concatenate([samples1, samples2])
    return torch.FloatTensor(samples).reshape(-1, 1)

def train_step(model, optimizer, real_samples):
    """one training step using contrastive divergence"""
    batch_size = len(real_samples)
    
    # sample some noise from a bit wider range than the data
    noise = torch.FloatTensor(batch_size, 1).uniform_(-4, 4)
    
    # get energies for real and noise samples
    real_energy = model(real_samples)
    noise_energy = model(noise)
    
    # push down real energy, push up noise energy
    loss = real_energy.mean() - noise_energy.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def visualize_model(model, real_data, title):
    """show what the model learned"""
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
    
    # samples from model using softmax sampling
    plt.subplot(133)
    with torch.no_grad():
        # generate candidate points
        candidates = torch.FloatTensor(50000, 1).uniform_(-10, 10)
        energies = model(candidates)
        
        # convert energies to probabilities with temperature
        temp = 0.1
        probs = torch.softmax(-energies/temp, dim=0)
        
        # sample 5k points according to probabilities
        idx = torch.multinomial(probs.view(-1), 5000, replacement=True)
        samples = candidates[idx]
        
        plt.hist(samples.numpy(), bins=50, density=True, alpha=0.5)
        plt.title('Sampled from Model')
        plt.xlabel('x')
        plt.ylabel('Density')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # setup
    real_data = generate_target_data(10000)
    model = SimpleEBM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # training settings
    n_epochs = 200
    print_every = 50
    
    # train the model
    for epoch in range(n_epochs):
        loss = train_step(model, optimizer, real_data)
        
        if (epoch + 1) % print_every == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}')
            visualize_model(model, real_data, f'EBM Training Progress - Epoch {epoch+1}')
