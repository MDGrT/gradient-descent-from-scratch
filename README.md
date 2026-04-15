# Gradient Descent from Scratch

Implemented batch, stochastic, and mini-batch gradient descent for linear regression using only NumPy.

## What I Learned
- Batch GD: Stable but slow per epoch
- SGD: Fast updates but noisy, benefits from learning rate scheduling
- Mini-batch: Best of both worlds, requires tuning batch size
- Feature scaling is non-negotiable for SGD convergence
- Scikit-learn's SGDRegressor uses L2 regularization and invscaling learning rate by default

## Key Insight
Linear regression is finding the projection of y onto the column space of X. The normal equation solves this in one step; gradient descent walks there iteratively.

## Files
- `batch-gradient-descent.ipynb`: Vectorized batch gradient descent
- `sgd.py`: Per-sample stochastic gradient descent
- `minibatch_gd.py`: Mini-batch with shuffling
- `comparison.ipynb`: Benchmarks against sklearn
