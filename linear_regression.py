import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Step 2: Initialize parameters
m = np.random.randn(1)  # Slope
b = np.random.randn(1)  # Intercept
learning_rate = 0.01
iterations = 1000

# Step 3: Gradient Descent
for i in range(iterations):
    y_pred = m * X + b  # Predicted y values
    m_grad = -2 * np.sum(X * (y - y_pred)) / len(y)  # Gradient for m
    b_grad = -2 * np.sum(y - y_pred) / len(y)  # Gradient for b
    m -= learning_rate * m_grad  # Update slope
    b -= learning_rate * b_grad  # Update intercept
    
    # Print MSE every 100 iterations
    if i % 100 == 0:
        mse = np.mean((y - y_pred) ** 2)
        print(f"Iteration {i}: MSE = {mse}")

# Step 4: Plotting the data and the
