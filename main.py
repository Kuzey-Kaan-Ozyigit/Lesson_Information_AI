import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

X = np.loadtxt("student_scores.txt", delimiter=',') / 100
y_pass_fail = np.loadtxt("pass_fail_labels.txt", dtype=int).reshape(-1, 1)
y_category = np.loadtxt("category_labels.txt", delimiter=',', dtype=int).tolist()

# Weights And Biases
np.random.seed(1)
input_layer_size = 3
hidden_layer_size = 6
hidden_layer_size_2 = 4
output_layer_size_1 = 1
output_layer_size_2 = 4

# Weights For The Hidden Layers
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size)
bias_hidden = np.zeros((1, hidden_layer_size))

weights_hidden_hidden = np.random.randn(hidden_layer_size, hidden_layer_size_2)
bias_hidden_2 = np.zeros((1, hidden_layer_size_2))

# Weights For The Output Layers
weights_hidden_output_1 = np.random.randn(hidden_layer_size_2, output_layer_size_1)
bias_output_1 = np.zeros((1, output_layer_size_1))

weights_hidden_output_2 = np.random.randn(hidden_layer_size_2, output_layer_size_2)
bias_output_2 = np.zeros((1, output_layer_size_2))

error_1_list = []
error_2_list = []

# Training
learning_rate = 0.1
for epoch in range(100000):
    # Forward Computation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    hidden_layer_input_2 = np.dot(hidden_layer_output, weights_hidden_hidden) + bias_hidden_2
    hidden_layer_output_2 = sigmoid(hidden_layer_input_2)

    final_input_1 = np.dot(hidden_layer_output_2, weights_hidden_output_1) + bias_output_1
    output_1 = sigmoid(final_input_1)  # Pass/Fail

    final_input_2 = np.dot(hidden_layer_output_2, weights_hidden_output_2) + bias_output_2
    output_2 = softmax(final_input_2)  # Category

    # Error Computations
    error_1 = y_pass_fail - output_1
    error_1_list.append(np.mean(np.abs(error_1)))
    error_2 = y_category - output_2
    error_2_list.append(np.mean(np.abs(error_2)))

    # Backpropagation
    d_output_1 = error_1 * sigmoid_derivative(output_1)
    d_output_2 = error_2

    d_hidden_layer_2_1 = d_output_1.dot(weights_hidden_output_1.T) * sigmoid_derivative(hidden_layer_output_2)
    d_hidden_layer_2_2 = d_output_2.dot(weights_hidden_output_2.T) * sigmoid_derivative(hidden_layer_output_2)

    d_hidden_layer_1 = (d_hidden_layer_2_1 + d_hidden_layer_2_2).dot(weights_hidden_hidden.T) * sigmoid_derivative(hidden_layer_output)

    # Update Weights
    weights_hidden_output_1 += hidden_layer_output_2.T.dot(d_output_1) * learning_rate
    bias_output_1 += np.sum(d_output_1, axis=0, keepdims=True) * learning_rate

    weights_hidden_output_2 += hidden_layer_output_2.T.dot(d_output_2) * learning_rate
    bias_output_2 += np.sum(d_output_2, axis=0, keepdims=True) * learning_rate

    weights_hidden_hidden += hidden_layer_output.T.dot(d_hidden_layer_2_1 + d_hidden_layer_2_2) * learning_rate
    bias_hidden_2 += np.sum(d_hidden_layer_2_1 + d_hidden_layer_2_2, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden_layer_1) * learning_rate
    bias_hidden += np.sum(d_hidden_layer_1, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        acc_pass_fail = np.mean((output_1 > 0.5) == y_pass_fail)
        print(f"Epoch {epoch}, Error1: {np.mean(np.abs(error_1))}, Error2: {np.mean(np.abs(error_2))}, Accuracy: {acc_pass_fail:.2f}")


# Outputs After Training
print("\nOutputs After Training (Pass/Fail):")
print(output_1)

print("\nOutputs After Training (Category):")
print(output_2)

# Drawing The Error Graph
plt.figure(figsize=(10, 5))

plt.plot(error_1_list, label="Error 1")
plt.plot(error_2_list, label="Error 2")

plt.title("Error Comparison")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.show()
