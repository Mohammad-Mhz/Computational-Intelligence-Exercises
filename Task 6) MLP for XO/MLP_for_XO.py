import tkinter as tk
import numpy as np


def press_button(button, row, column):
    if button_states[row, column] == -1:
        button.config(bg='purple')
        button_states[row, column] = 1
    else:
        button.config(bg='yellow')
        button_states[row, column] = -1


def create_button(root, row, column):
    buttons = np.empty((row, column), dtype=tk.Button)
    for i in range(row):
        for j in range(column):
            button = tk.Button(root, text='      ', bg='yellow')
            button.config(command=lambda b=button, r=i, c=j: press_button(b, r, c))
            button.grid(row=i, column=j, padx=10, pady=10)
            buttons[i, j] = button
    return buttons


def check():
    flatten_input = button_states.flatten()
    return flatten_input


def read_matrix(file_path):
    matrices = []
    with open(file_path, 'r') as file:
        matrix_row = []
        for line in file:
            line = line.strip()
            if line:
                matrix_row.append(line)
            elif matrix_row:
                matrices.append(matrix_row)
                matrix_row = []
    return matrices


def flatten(in_matrix):
    flattened_matrix = []
    for row in in_matrix:
        flattened_matrix.extend(map(int, row.split()))
    return flattened_matrix


def sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1


def sigmoid_derivative(x):
    return (2 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))


# Load and prepare data
matrices_x = read_matrix('matrix_x.txt')
matrices_o = read_matrix('matrix_o.txt')

arrays_x = [flatten(matrix) for matrix in matrices_x]
arrays_o = [flatten(matrix) for matrix in matrices_o]

arrays = arrays_x + arrays_o

y_x = [1] * len(arrays_x)
y_o = [-1] * len(arrays_o)

y_total = y_x + y_o

# Adding bias to inputs
bias = 1
for array in arrays:
    array.append(bias)

inputs = np.array(arrays)
targets = np.array(y_total).reshape(-1, 1)

# Initialize weights
input_layer_size = inputs.shape[1]
hidden_layer_size = 10
output_layer_size = 1

weights_input_hidden = np.random.uniform(size=(input_layer_size, hidden_layer_size))
weights_hidden_output = np.random.uniform(size=(hidden_layer_size, output_layer_size))

learning_rate = 0.1
error_threshold = 0.0001

# Training the MLP
epoch = 0
while True:
    epoch += 1
    # Forward propagation
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_input = np.dot(hidden_layer_output, weights_hidden_output)
    final_output = sigmoid(final_input)

    # Calculate error
    error = targets - final_output
    mean_squared_error = np.mean(np.square(error))

    if mean_squared_error < error_threshold:
        print(f"Training stopped after {epoch} epochs")
        break

    # Backward propagation
    d_final_output = error * sigmoid_derivative(final_output)

    error_hidden_layer = d_final_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating weights
    weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate


def main():
    global button_states

    root = tk.Tk()
    root.title("Button Matrix")
    root.geometry("400x350")

    rows = 5
    cols = 5
    button_states = (np.ones((rows, cols), dtype=int)) * (-1)
    buttons = create_button(root, rows, cols)

    def submit():
        flattened_array = check()
        test_input = np.append(flattened_array, 1)

        hidden_layer_input = np.dot(test_input, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        final_input = np.dot(hidden_layer_output, weights_hidden_output)
        result = sigmoid(final_input)

        if result > 0:
            output = "X"
        else:
            output = "O"

        output_label = tk.Label(root, text=f'This is {output}', font=('arial', 20, 'bold'))
        output_label.grid(row=7, column=7, pady=20, padx=20)

    submit_button = tk.Button(root, text="Check", bg='green', command=submit)
    submit_button.grid(row=rows, columnspan=cols, pady=10)
    root.mainloop()


if __name__ == "__main__":
    main()
