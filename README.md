# Energy-based Model Demo

A minimal example showing how energy-based models work. This code learns a simple 1D two-bump distribution using an energy-based model (EBM).

The model learns to assign low energy to real data points and high energy to fake/noise points. It's trained using contrastive divergence - basically pushing down energy for real samples while pushing up energy for random noise.

The visualization shows:

- The learned energy landscape
- The actual data distribution (two Gaussian bumps)
- Samples generated from the model using softmax sampling
