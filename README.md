# Neural Network for Binary and Categorical Classification

This project implements a neural network (NN) model to perform two distinct tasks:
1. **Pass/Fail Classification**.
2. **Categorical Classification** (multi-class).

## Files
- `main.py`: Project source code.
- `student_scores.txt`: Student score dataset.
- `pass_fail_labels.txt`: Pass/Fail labels.
- `category_labels.txt`: Category labels.

## Libraries Used
- `numpy`: For mathematical operations.
- `pandas`: For data manipulation.
- `matplotlib`: For plotting error graphs.

## Model Architecture
- **Input Layer**: 3 neurons (for student scores input).
- **Hidden Layer 1**: 6 neurons.
- **Hidden Layer 2**: 4 neurons.
- **Output Layers**:
  - Output 1: Pass/Fail classification (1 neuron, Sigmoid activation function).
  - Output 2: Categorical classification (4 neurons, Softmax activation function).

## Training Process
- **Activation Functions**:
  - `sigmoid`: For Pass/Fail output.
  - `softmax`: For categorical classification output.
- **Loss Function**:
  - Mean Absolute Error (MAE).
- **Optimization**:
  - Backpropagation algorithm.
  - Learning rate: `0.1`.
- **Epochs**: 100,000.

## Usage
1. Install the required libraries.
2. Ensure `student_scores.txt`, `pass_fail_labels.txt`, and `category_labels.txt` files are in the correct format.
3. Run the `main.py` script to train the model.

## Outputs
- During training, error graphs (`Error 1` and `Error 2`) are plotted.
- After training, classification results are displayed in the console.

## Graph
- The **Error Comparison** graph shows how the errors for both tasks evolve over the training epochs.

## Notes
This project requires properly formatted datasets and the necessary dependencies to run correctly.
