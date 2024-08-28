import tkinter as tk
import numpy as np


def press_button(button, row, colum):
    if button_states[row, colum] == -1:
        button.config(bg='purple')
        button_states[row, colum] = 1
    else:
        button.config(bg='yellow')
        button_states[row, colum] = -1


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
                # print(matrix_row)
                matrix_row = []
    # print(matrices)
    return matrices


def flatten(in_matrix):
    flattened_matrix = []

    for row in in_matrix:
        # print(row.split())
        flattened_matrix.extend(map(int, row.split()))
    # print(flattened_matrix)
    return flattened_matrix


def activation_function(input, theta):
    if input > theta:
        y = 1
    elif input < -theta:
        y = -1
    else:
        y = 0

    return y


matrices_x = read_matrix('matrix_x.txt')
matrices_o = read_matrix('matrix_o.txt')

arrays_x = [flatten(matrix) for matrix in matrices_x]
arrays_o = [flatten(matrix) for matrix in matrices_o]

arrays = arrays_x + arrays_o
# Targets
y_x = []
for array in arrays_x:
    y_x.append(1)

y_o = []
for array in arrays_o:
    y_o.append(-1)

y_total = y_x + y_o

bias = 1
for array in arrays:  # Appending bias to inputs
    array.append(bias)

inputs = np.array(arrays)
targets = np.array(y_total)
weights = np.zeros(26, int)
a = 1  # alpha - learning rate
theta = 0.2

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
        result = np.dot(test_input, weights)
        if result > theta:
            output = "X"
        elif result < -theta:
            output = "O"
        else:
            output = "Unknown"
        output_label = tk.Label(root, text=f'This is {output}', font=('arial', 20, 'bold'))
        output_label.grid(row=7, column=7, pady=20, padx=20)

    submit_button = tk.Button(root, text="Check", bg='green', command=submit)
    submit_button.grid(row=rows, columnspan=cols, pady=10)
    root.mainloop()


if __name__ == "__main__":
    main()
