import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data for binary classification
np.random.seed(0)
X = np.random.randn(100, 2)  # 100 data points with 2 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Class 1 if the sum of the features > 0, else Class 0

# Step 2: Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 3: Initialize parameters
m, n = X.shape
weights = np.random.randn(n)
bias = 0
learning_rate = 0.1
iterations = 1000

# Step 4: Gradient Descent
for i in range(iterations):
    # Calculate linear model and predictions
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    
    # Compute gradients
    dw = (1 / m) * np.dot(X.T, (y_pred - y))  # Gradient for weights
    db = (1 / m) * np.sum(y_pred - y)         # Gradient for bias
    
    # Update weights and bias
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    # Print the loss every 100 iterations
    if i % 100 == 0:
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        print(f"Iteration {i}: Loss = {loss}")

# Step 5: Plot the data and decision boundary
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

# Calculate and plot decision boundary
x_boundary = np.array([min(X[:, 0]), max(X[:, 0])])
y_boundary = -(weights[0] * x_boundary + bias) / weights[1]
plt.plot(x_boundary, y_boundary, color='green', label='Decision Boundary')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Final weights and bias
print("Final weights:", weights)
print("Final bias:", bias)
