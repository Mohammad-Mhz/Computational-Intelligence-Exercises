import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return (1 - (1 / (1 + np.exp(-x)))) * (1 / (1 + np.exp(-x)))


dataframe1 = pd.read_excel('thyroidInputs.xlsx')  # inputs
dataframe2 = pd.read_excel('thyroidTargets.xlsx')  # targets

data_array1 = dataframe1.to_numpy()
data_array2 = dataframe2.to_numpy()

inputs = data_array1.T
targets = data_array2.T

# Split the dataset
train_inputs, temp_inputs, train_targets, temp_targets = train_test_split(inputs, targets, test_size=0.3,
                                                                          random_state=42)
val_inputs, test_inputs, val_targets, test_targets = train_test_split(temp_inputs, temp_targets, test_size=0.5,
                                                                      random_state=42)

input_layer_size = inputs.shape[1]
hidden_layer_size = 10
output_layer_size = targets.shape[1]
# print(input_layer_size, hidden_layer_size, output_layer_size)

weights_input_hidden = np.random.uniform(0.1, 0.5, size=(input_layer_size, hidden_layer_size))
weights_hidden_output = np.random.uniform(0.1, 0.5, size=(hidden_layer_size, output_layer_size))
# print(weights_input_hidden.shape, weights_hidden_output.shape)
# print(train_inputs.shape)

learning_rate = 0.1
error_threshold = 0.005
max_epochs = 1000

# Training the MLP
epoch = 0
train_mse = []
val_mse = []
while epoch < max_epochs:
    epoch += 1

    epoch_train_mse = []

    for i in range(len(train_inputs)):
        # Forward propagation
        hidden_layer_input = np.dot(train_inputs[i], weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        final_input = np.dot(hidden_layer_output, weights_hidden_output)
        final_output = sigmoid(final_input)

        # Calculate error
        error = train_targets[i] - final_output
        train_mean_squared_error = np.mean(np.square(error))
        epoch_train_mse.append(train_mean_squared_error)

        # Backward propagation
        d_final_output = error * sigmoid_derivative(final_input)
        error_hidden_layer = d_final_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_input)

        # Updating weights
        weights_hidden_output += np.outer(hidden_layer_output, d_final_output) * learning_rate
        weights_input_hidden += np.outer(train_inputs[i], d_hidden_layer) * learning_rate

    train_mse.append(np.mean(epoch_train_mse))

    if train_mse[-1] < error_threshold:
        print(f"Training stopped after {epoch} epochs with MSE: {train_mse[-1]}")
        break

    # Forward propagation on validation set
    hidden_layer_input_val = np.dot(val_inputs, weights_input_hidden)
    hidden_layer_output_val = sigmoid(hidden_layer_input_val)

    final_input_val = np.dot(hidden_layer_output_val, weights_hidden_output)
    final_output_val = sigmoid(final_input_val)

    # Calculate error on validation set
    val_error = val_targets - final_output_val
    val_mean_squared_error = np.mean(np.square(val_error))
    val_mse.append(val_mean_squared_error)

    if epoch == max_epochs:
        print(
            f"Training stopped after reaching the maximum number of epochs ({max_epochs}) with MSE: {train_mean_squared_error}")


# print(weights_hidden_output)
# print(weights_input_hidden)

# Helper function to compute predictions
def predict(inputs, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_input = np.dot(hidden_layer_output, weights_hidden_output)
    final_output = sigmoid(final_input)
    # Get predicted classes
    predicted_classes = np.argmax(final_output, axis=1)
    return predicted_classes


class_names = ['Normal', 'Hyperfunc', 'Subnormalfunc']
# Compute predictions
train_predictions = predict(train_inputs, weights_input_hidden, weights_hidden_output)
val_predictions = predict(val_inputs, weights_input_hidden, weights_hidden_output)
test_predictions = predict(test_inputs, weights_input_hidden, weights_hidden_output)

# Compute confusion matrices
train_conf_matrix = confusion_matrix(np.argmax(train_targets, axis=1), train_predictions)
val_conf_matrix = confusion_matrix(np.argmax(val_targets, axis=1), val_predictions)
test_conf_matrix = confusion_matrix(np.argmax(test_targets, axis=1), test_predictions)


# Plot confusion matrices
def plot_confusion_matrix(conf_matrix, title, class_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


# Plot confusion matrices for training, validation, and test sets
plot_confusion_matrix(train_conf_matrix, "Training Set Confusion Matrix", class_names)
plot_confusion_matrix(val_conf_matrix, "Validation Set Confusion Matrix", class_names)
plot_confusion_matrix(test_conf_matrix, "Test Set Confusion Matrix", class_names)
