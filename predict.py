import numpy as np

while True:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # Eğitilmiş parametreleri yükle
    weights_input_hidden = np.load('trained_weights_input_hidden.npy')
    bias_hidden = np.load('trained_bias_hidden.npy')
    weights_hidden_hidden = np.load('trained_weights_hidden_hidden.npy')
    bias_hidden_2 = np.load('trained_bias_hidden_2.npy')
    weights_hidden_output_1 = np.load('trained_weights_hidden_output_1.npy')
    bias_output_1 = np.load('trained_bias_output_1.npy')
    weights_hidden_output_2 = np.load('trained_weights_hidden_output_2.npy')
    bias_output_2 = np.load('trained_bias_output_2.npy')

    def predict(scores):
        # Girdiyi normalize et ve hazırla
        X = np.array(scores) / 100
        X = X.reshape(1, -1)  # Batch boyutu ekle
        
        # Forward pass
        hidden_layer_output = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden)
        hidden_layer_output_2 = sigmoid(np.dot(hidden_layer_output, weights_hidden_hidden) + bias_hidden_2)
        
        # Pass/Fail tahmini
        pass_fail_output = sigmoid(np.dot(hidden_layer_output_2, weights_hidden_output_1) + bias_output_1)
        
        # Kategori tahmini
        category_output = softmax(np.dot(hidden_layer_output_2, weights_hidden_output_2) + bias_output_2)
        
        return pass_fail_output, category_output

    # Kullanıcı girdisi
    test_scores = list(map(float, input("3 ders notunu girin (virgülle ayırın): ").split(',')))
    pass_fail, category = predict(test_scores)

    print(f"\nPass/Fail Olasılığı: {pass_fail[0][0]:.4f}")
    print(f"Tahmin Edilen Kategori: {np.argmax(category)}. sınıf")
    print("Kategori Dağılımı:")
    for i, prob in enumerate(category[0]):
        print(f"{i}. sınıf: {prob:.4f}")