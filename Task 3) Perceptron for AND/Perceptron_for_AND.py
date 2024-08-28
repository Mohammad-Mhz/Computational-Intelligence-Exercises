import numpy as np
import matplotlib.pyplot as plt

weights = np.array([0, 0, 0])  # [w1, w2, bias]
inputs = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1]])
targets = np.array([1, -1, -1, -1])
a = 1  # alpha - learning rate
theta = 0.2


def plot_decision_boundary(weights):
    x = np.linspace(-2, 2, 100)
    y = -(weights[0] * x + weights[2]) / weights[1]
    y2 = -(weights[0] * x + weights[2] - theta) / weights[1]
    y3 = -(weights[0] * x + weights[2] + theta) / weights[1]
    plt.plot(x, y, label='Decision Boundary')
    plt.plot(x, y2, label='+ theta')
    plt.plot(x, y3, label='- theta')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.legend()


def activation_function(input, theta):
    if input > theta:
        y = 1
    elif input < -theta:
        y = -1
    else:
        y = 0

    return y


condition = False
while not condition:
    condition = True
    for i in range(len(inputs)):
        Y_net = np.dot(inputs[i], weights)
        # print(Y_net)
        output = activation_function(Y_net, theta)

        if output != targets[i]:
            weights += inputs[i] * targets[i] * a
            condition = False

        print(weights)

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets, cmap='coolwarm', marker='o')
plot_decision_boundary(weights)

plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

while True:

    input_1 = float(input("Enter your first input: "))
    input_2 = float(input("Enter your second input: "))

    test_input = np.array([input_1, input_2, 1])  # The third input is bias

    result = np.dot(test_input, weights)

    if result > theta:
        output = 1
    elif result < -theta:
        output = -1
    else:
        output = 0

    print("Output for input", test_input, ":", output)

    continue_input = input("Do you want to enter another input? (y/n): ").strip().lower()
    if continue_input not in ['y', 'yes']:
        break
