import numpy as np

while True:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # Install the Trained Parameters
    weights_input_hidden = np.load('trained_weights_input_hidden.npy')
    bias_hidden = np.load('trained_bias_hidden.npy')
    weights_hidden_hidden = np.load('trained_weights_hidden_hidden.npy')
    bias_hidden_2 = np.load('trained_bias_hidden_2.npy')
    weights_hidden_output_1 = np.load('trained_weights_hidden_output_1.npy')
    bias_output_1 = np.load('trained_bias_output_1.npy')
    weights_hidden_output_2 = np.load('trained_weights_hidden_output_2.npy')
    bias_output_2 = np.load('trained_bias_output_2.npy')

    def predict(scores):
        # Normalize the Input
        X = np.array(scores) / 100
        X = X.reshape(1, -1)
        
        # Forward Pass
        hidden_layer_output = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden)
        hidden_layer_output_2 = sigmoid(np.dot(hidden_layer_output, weights_hidden_hidden) + bias_hidden_2)
        
        # Pass/Fail Prediction
        pass_fail_output = sigmoid(np.dot(hidden_layer_output_2, weights_hidden_output_1) + bias_output_1)
        
        # Category Prediction
        category_output = softmax(np.dot(hidden_layer_output_2, weights_hidden_output_2) + bias_output_2)
        
        return pass_fail_output, category_output

    # User Input
    test_scores = list(map(float, input("Enter the 3 lessons score(Separate with a comma and don't use space): ").split(',')))
    pass_fail, category = predict(test_scores)

    if pass_fail >= 0.5:
        pass_fail = "Pass"
    else:
        pass_fail = "Fail"
    print(f"\nPass/Fail: {pass_fail}")
    print(f"Predicted Category: Category {np.argmax(category)}")
    print("Categories:")
    for i, prob in enumerate(category[0]):
        print(f"Category {i}: {prob:.4f}")
    print("P.S.\nCategory 0 = Poor\nCategory 1 = Average\nCategory 2 = Good\nCategory 3 = Excellent")
