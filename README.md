# Energy-based Model Demo

EBM model that identifies real vs fake data points, trained using contrastive divergence - basically pushing down energy for real samples while pushing up energy for random noise.

viz shows:

- The learned energy landscape
- The actual data distribution (two Gaussian bumps, ring, or spiral)
- Samples generated from the model using softmax / temperature (to see what the model thinks is likely realistic from a bucket of random)

using softplus is critical here because we can learn from any high-energy data while getting really good at low-energy data
tanh saturates quickly at extremes so it doesnt really learn as much from fake data i think, gradient becomes zero for anything that looks fake
need one of the activations because otherwise totally unstable, it just learns to hyper maximize low energy for the real data

# Sequential
similar but uses mean pooling... trying to think of a simple example that breaks the mean pooling 
