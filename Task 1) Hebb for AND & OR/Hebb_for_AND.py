import numpy as np
import matplotlib.pyplot as plt

weights = np.array([0, 0, 0])  # [w1, w2, bias]
X = np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]])
y = np.array([1, -1, -1, -1])


# Plotting function for decision boundary
def plot_decision_boundary(weights):
    x = np.linspace(-2, 2, 100)
    y = -(weights[0] * x + weights[2]) / weights[1]
    plt.plot(x, y, label='Decision Boundary')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.legend()


# Plot initial data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o')
plt.title('Initial Data Points')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Training loop
for i in range(len(X)):
    # Update weights_ using Hebb-Net rule
    weights += X[i] * y[i]
    print(weights)

    # Plot decision boundary at each step
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o')
    plot_decision_boundary(weights)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# Testing
while True:

    input_1 = int(input("Enter your first input: "))
    input_2 = int(input("Enter your second input: "))

    test_input = np.array([input_1, input_2, 1])  # The third input is bias

    result = np.dot(test_input, weights)

    output = 1 if result >= 0 else -1

    print("Output for input", test_input, ":", output)

    continue_input = input("Do you want to enter another input? (y/n): ").strip().lower()
    if continue_input not in ['y', 'yes']:
        break
