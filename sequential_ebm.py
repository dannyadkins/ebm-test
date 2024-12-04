import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class SequentialEBM(nn.Module):
    """model that learns patterns in sets of points"""
    def __init__(self, point_dim=2, hidden_dims=[64, 64]):
        super().__init__()
        
        # process each point through same network first
        point_layers = []
        prev_dim = point_dim
        for h_dim in hidden_dims:
            point_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
            
        self.point_net = nn.Sequential(*point_layers)
        
        # process the combined features
        self.energy_net = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.ReLU(),
            nn.Linear(prev_dim, 1)
        )
        
    def forward(self, x):
        # x has shape (batch_size, n_points, point_dim)
        batch_size = x.shape[0]
        
        # run each point through the network
        point_features = self.point_net(x.reshape(-1, x.shape[-1]))
        point_features = point_features.reshape(batch_size, -1, point_features.shape[-1])
        
        # average the point features
        set_features = point_features.mean(dim=1)
        
        # get final energy score
        energy = self.energy_net(set_features)
        return energy

def generate_circle_points(n_samples=1000, points_per_set=30):
    """make sets of points in circles with random centers and sizes"""
    sets = []
    for _ in range(n_samples):
        # pick random circle properties
        center = np.random.uniform(-4, 4, 2)
        radius = np.random.uniform(0.5, 2)
        
        # make points around the circle
        angles = np.random.uniform(0, 2*np.pi, points_per_set)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        points = np.stack([x, y], axis=1)
        
        # add a bit of noise
        points += np.random.normal(0, 0.05, points.shape)
        sets.append(points)
        
    return torch.FloatTensor(np.stack(sets))

def generate_spiral_points(n_samples=1000, points_per_set=30):
    """make sets of points in spiral shapes"""
    sets = []
    for _ in range(n_samples):
        # pick random spiral properties
        center = np.random.uniform(-4, 4, 2)
        scale = np.random.uniform(0.5, 2)
        rotation = np.random.uniform(0, 2*np.pi)
        
        # make points along the spiral
        t = np.linspace(0, 4*np.pi, points_per_set)
        r = t/4
        x = center[0] + scale * (r * np.cos(t + rotation))
        y = center[1] + scale * (r * np.sin(t + rotation))
        points = np.stack([x, y], axis=1)
        
        # add a bit of noise
        points += np.random.normal(0, 0.05, points.shape)
        sets.append(points)
        
    return torch.FloatTensor(np.stack(sets))

def generate_line_points(n_samples=1000, points_per_set=30):
    """make sets of points in straight lines"""
    sets = []
    for _ in range(n_samples):
        # pick random line properties
        start = np.random.uniform(-4, 4, 2)
        angle = np.random.uniform(0, 2*np.pi)
        length = np.random.uniform(2, 4)
        
        # make points along the line
        t = np.linspace(0, 1, points_per_set)
        end = start + length * np.array([np.cos(angle), np.sin(angle)])
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        points = np.stack([x, y], axis=1)
        
        # add a bit of noise
        points += np.random.normal(0, 0.05, points.shape)
        sets.append(points)
        
    return torch.FloatTensor(np.stack(sets))

def generate_bimodal_points(n_samples=1000, points_per_set=30):
    """make sets of points in two clusters - this is tricky since the average point would be between clusters"""
    sets = []
    for _ in range(n_samples):
        # make two cluster centers
        center1 = np.random.uniform(-4, 4, 2)
        center2 = center1 + np.random.uniform(2, 3, 2)
        
        # split points between clusters
        n_points1 = points_per_set // 2
        n_points2 = points_per_set - n_points1
        
        # make points around first cluster
        points1 = np.random.normal(0, 0.2, (n_points1, 2)) + center1
        
        # make points around second cluster
        points2 = np.random.normal(0, 0.2, (n_points2, 2)) + center2
        
        # put the points together
        points = np.vstack([points1, points2])
        sets.append(points)
        
    return torch.FloatTensor(np.stack(sets))

def generate_multimodal_points(n_samples=1000, points_per_set=30):
    """make sets of points in multiple clusters with varying densities - this breaks mean pooling since
    the model can't capture the relative densities and spatial relationships between clusters"""
    sets = []
    for _ in range(n_samples):
        # randomly choose number of clusters (2-5)
        n_clusters = 10
        
        # generate cluster centers with minimum spacing
        centers = []
        densities = []
        min_spacing = 2.0
        
        # first cluster
        centers.append(np.random.uniform(-4, 4, 2))
        densities.append(np.random.uniform(0.1, 0.4))
        
        # add more clusters with spacing constraints
        for _ in range(n_clusters-1):
            while True:
                new_center = np.random.uniform(-4, 4, 2)
                if all(np.linalg.norm(new_center - c) > min_spacing for c in centers):
                    centers.append(new_center)
                    densities.append(np.random.uniform(0.1, 0.4))
                    break
        
        # normalize densities to sum to 1
        densities = np.array(densities) / sum(densities)
        
        # assign points to clusters based on densities
        points = []
        remaining_points = points_per_set
        for i in range(n_clusters):
            if i == n_clusters - 1:
                n_cluster_points = remaining_points
            else:
                n_cluster_points = int(points_per_set * densities[i])
                remaining_points -= n_cluster_points
            
            # generate points for this cluster
            cluster_points = np.random.normal(0, 0.2, (n_cluster_points, 2)) + centers[i]
            points.append(cluster_points)
        
        # combine all points
        points = np.vstack(points)
        sets.append(points)
        
    return torch.FloatTensor(np.stack(sets))

# get points based on pattern type
pattern_funcs = {
    'circle': generate_circle_points,
    'spiral': generate_spiral_points,
    'bimodal': generate_bimodal_points,
    'line': generate_line_points,
    'multimodal': generate_multimodal_points
}

def generate_random_points(n_samples=1000, points_per_set=30, pattern='multimodal'):
    """Generate fake data by perturbing real data patterns"""
    real_sets = pattern_funcs.get(pattern, generate_line_points)(n_samples, points_per_set)
    
    # add  noise to create fake data
    # scale noise by the std of the real data to maintain relative scale
    noise_scale = 0.5 * torch.std(real_sets)
    noise = torch.randn_like(real_sets) * noise_scale
    
    # also randomly permute some of the points to break patterns
    fake_sets = real_sets + noise
    
    # vec permutation instead of loop
    should_permute = torch.rand(len(fake_sets)) < 0.2
    perms = torch.stack([torch.randperm(points_per_set) for _ in range(should_permute.sum())])
    fake_sets[should_permute] = fake_sets[should_permute].gather(1, perms.unsqueeze(-1).expand(-1, -1, 2))
            
    return fake_sets

def train_step(model, optimizer, real_sets):
    """do one training step"""
    batch_size = len(real_sets)
    
    # make some random noise sets
    noise_sets = generate_random_points(batch_size, real_sets.shape[1])
    
    # get energy scores
    real_energy = model(real_sets)
    noise_energy = model(noise_sets)
    
    # make real energy low and noise energy high
    loss = torch.nn.functional.softplus(real_energy).mean() + \
           torch.nn.functional.softplus(-noise_energy).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def visualize_model(model, epoch, pattern='circle'):
    """show what the model has learned"""
    plt.figure(figsize=(15, 5))
    
    # show some real examples
    plt.subplot(131)
    

    real_sets = pattern_funcs.get(pattern, generate_line_points)(5)
    title = f"Real {pattern.title()} Point Sets"
        
    for points in real_sets:
        plt.scatter(points[:,0], points[:,1], alpha=0.5)
    plt.title(title)
    plt.axis('equal')

    # show energy landscape
    plt.subplot(133)
    with torch.no_grad():
        # mix random and pattern points
        n_test_sets = 50000
        n_real = n_test_sets // 2
        n_random = n_test_sets - n_real
        
        # get both types of points
        random_sets = generate_random_points(n_random, points_per_set=30)

        # whether to mix in real data 
        mix_real = True 

        if mix_real:
            real_sets = pattern_funcs.get(pattern, generate_line_points)(n_real)
            test_sets = torch.cat([random_sets, real_sets], dim=0)
        else:
            test_sets = random_sets
            
        # find best and worst sets
        test_energies = model(test_sets)
        
        min_idx = torch.argmin(test_energies)
        max_idx = torch.argmax(test_energies)
        
        low_energy_set = test_sets[min_idx:min_idx+1]
        high_energy_set = test_sets[max_idx:max_idx+1]
        
        # make energy grid using best set as reference
        x = np.linspace(-6, 6, 50)
        y = np.linspace(-6, 6, 50)
        X, Y = np.meshgrid(x, y)
        
        energies = []
        for i in range(len(x)):
            row_energies = []
            for j in range(len(y)):
                test_point = torch.FloatTensor([[[X[i,j], Y[i,j]]]])
                points = torch.cat([low_energy_set[:,:-1], test_point], dim=1)
                energy = model(points).item()
                row_energies.append(energy)
            energies.append(row_energies)
            
        
        # show best and worst sets
        plt.scatter(low_energy_set[0,:,0], low_energy_set[0,:,1], c='green', alpha=0.5,
                   label=f'Low Energy Set')
        plt.scatter(high_energy_set[0,:,0], high_energy_set[0,:,1], c='red', alpha=0.5,
                   label=f'High Energy Set')
        plt.legend()
        plt.title("Energy Landscape with\nBest and Worst Sets")
    
    plt.suptitle(f'Training Progress - Epoch {epoch}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # setup training
    n_epochs = 5000
    batch_size = 100
    vis_every = 1000
    pattern = 'multimodal'  # can be circle, spiral, line, bimodal, or multimodal
    
    # create model
    model = SequentialEBM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # train the model
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        # get training data
        real_sets = pattern_funcs.get(pattern, generate_line_points)(batch_size)
        
        # do training step
        loss = train_step(model, optimizer, real_sets)
        pbar.set_description(f"Loss: {loss:.4f}")
        
        if (epoch + 1) % vis_every == 0:
            visualize_model(model, epoch + 1, pattern)
