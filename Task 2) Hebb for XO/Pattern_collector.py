import tkinter as tk  # Also we can use kivy
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
            button = tk.Button(root, text=f'button{i}_{j}', bg='yellow')
            button.config(command=lambda b=button, r=i, c=j: press_button(b, r, c))
            button.grid(row=i, column=j, padx=10, pady=10)
            buttons[i, j] = button
    return buttons


counter = 0


def save_matrix():
    global counter
    with open("matrix_o.txt", "a") as file:  # Use "a" mode to append to the file
        for row in button_states:
            file.write(" ".join(map(str, row)) + "\n")
        file.write("\n")

    print(f"Matrix {counter} saved.")
    counter += 1


def main():
    global button_states

    root = tk.Tk()
    root.title("Button Matrix")
    rows = 5
    cols = 5
    button_states = (np.ones((rows, cols), dtype=int)) * (-1)
    buttons = create_button(root, rows, cols)

    submit_button = tk.Button(root, text="Submit", command=save_matrix)
    submit_button.grid(row=rows, columnspan=cols, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
