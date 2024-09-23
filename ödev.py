import numpy as np
import random

random.seed(2)

# Sigmoid aktivasyon fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid aktivasyon fonksiyonunun türevi
def sigmoid_derivative(x):
    return x * (1 - x)

# Yapay sinir ağı sınıfı
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Ağırlıkları rastgele başlat
        self.weights_input_hidden1 = np.random.rand(input_size, hidden_size1)
        self.weights_hidden1_hidden2 = np.random.rand(hidden_size1, hidden_size2)
        self.weights_hidden2_output = np.random.rand(hidden_size2, output_size)

    def forward(self, inputs):
        # İleri yayılım (forward propagation)
        self.hidden1_input = np.dot(inputs, self.weights_input_hidden1)
        self.hidden1_output = sigmoid(self.hidden1_input)

        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2)
        self.hidden2_output = sigmoid(self.hidden2_input)

        self.output_layer_input = np.dot(self.hidden2_output, self.weights_hidden2_output)
        self.output = sigmoid(self.output_layer_input)

        return self.output

    def backward(self, inputs, targets, learning_rate):
        # Geri yayılım (backward propagation)
        # Hata hesaplama
        output_error = targets - self.output

        # Çıkış katmanındaki hata ve sigmoid türevi kullanarak delta hesaplama
        output_delta = output_error * sigmoid_derivative(self.output)

        # Gizli katmanlardaki hata ve sigmoid türevi kullanarak delta hesaplama
        hidden2_error = output_delta.dot(self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * sigmoid_derivative(self.hidden2_output)

        hidden1_error = hidden2_delta.dot(self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * sigmoid_derivative(self.hidden1_output)

        # Ağırlıkları güncelleme
        self.weights_hidden2_output += self.hidden2_output.T.dot(output_delta) * learning_rate
        self.weights_hidden1_hidden2 += self.hidden1_output.T.dot(hidden2_delta) * learning_rate
        self.weights_input_hidden1 += inputs.T.dot(hidden1_delta) * learning_rate

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            # Ağı eğitme
            for i in range(len(inputs)):
                input_data = np.array([inputs[i]])
                target_data = np.array([targets[i]])

                # İleri ve geri yayılım
                output = self.forward(input_data)
                self.backward(input_data, target_data, 1 - (epoch/epochs))

                # Belirlenen epoch'ta bir hata kontrolü, training'de kullanılan 70 tanesinin hatası
                if epoch % 10000 == 0:
                    error = np.mean(np.abs(target_data - output))
                    print(f"Epoch {epoch}, Error: {error}\n")

input_size = 2
hidden_size1 = 4
hidden_size2 = 3
output_size = 1

nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

# Eğitim verileri
inputs = np.array([[0.3,1.0],[0.3,0.2],[0.3,0.1],[0.1,0.3],[0.9,0.8],[0.9,0.2],[0.1,0.4],[1.0,0.3],[0.4,0.2],[0.2,0.2],[0.5,0.6],[0.7,0.6],[0.4,0.6],[0.9,0.6],[0.8,0.8],[1.0,0.7],[0.7,0.3],[1.0,0.1],[0.3,0.9],[1.0,0.6],[0.7,0.8],[0.2,1.0],[0.9,1.0],[0.1,0.8],[0.7,1.0],[0.8,0.7],[0.3,0.4],[0.7,0.4],[0.4,0.1],[0.5,0.3],[0.6,0.5],[0.3,0.5],[0.2,0.4],[1.0,0.9],[0.2,0.3],[0.2,0.9],[0.6,0.8],[0.5,0.9],[0.3,0.3],[0.7,0.2],[0.6,0.1],[0.6,0.3],[0.8,0.3],[0.4,0.3],[0.8,0.9],[0.8,0.4],[1.0,0.4],[0.3,0.7],[0.5,1.0],[0.2,0.8],[0.3,0.8],[0.6,0.9],[0.2,0.1],[0.8,0.6],[0.1,0.9],[0.1,1.0],[0.9,0.1],[0.8,0.5],[0.9,0.9],[0.1,0.7],[0.4,0.5],[0.7,0.5],[0.8,0.2],[0.5,0.8],[1.0,1.0],[0.2,0.5],[0.7,0.1],[0.9,0.4],[0.4,0.9],[0.1,0.6]])
targets = np.array([[0.3],[0.06],[0.03],[0.03],[0.72],[0.18],[0.04],[0.3],[0.08],[0.04],[0.3],[0.42],[0.24],[0.54],[0.64],[0.7],[0.21],[0.1],[0.27],[0.6],[0.56],[0.2],[0.9],[0.08],[0.7],[0.56],[0.12],[0.28],[0.04],[0.15],[0.3],[0.15],[0.08],[0.9],[0.06],[0.18],[0.48],[0.45],[0.09],[0.14],[0.06],[0.18],[0.24],[0.12],[0.72],[0.32],[0.4],[0.21],[0.5],[0.16],[0.24],[0.54],[0.02],[0.48],[0.09],[0.1],[0.09],[0.4],[0.81],[0.07],[0.2],[0.35],[0.16],[0.4],[1.0],[0.1],[0.07],[0.36],[0.36],[0.06]])

# Ağı eğitme
nn.train(inputs, targets, epochs= 50000)

# Test verisi
test_input = np.array([[0.9,0.7],[0.5,0.7],[0.5,0.1],[0.5,0.2],[0.4,0.8],[0.9,0.3],[0.6,0.4],[0.1,0.5],[1,0.5],[1,0.2],[1,0.8],[0.6,0.6],[0.3,0.6],[0.4,0.7],[0.6,0.7],[0.4,1],[0.9,0.5],[0.7,0.7],[0.9,0.9],[0.6,1],[0.6,0.2],[8,0.1],[0.1,0.2],[0.2,0.7],[0.1,0.1],[0.8,0.1],[0.5,0.5],[0.5,0.4],[0.2,0.6],[0.4,0.4]])
test_output = np.array([63,35,5,10,32,27,24,5,50,20,80,36,18,28,42,40,45,49,81,60,12,80,2,14,1,8,25,20,12,16,])
result = nn.forward(test_input)
normalize_result = result * 100 # Normalize etme

for i in range(len(test_input)):
    test_error = np.mean(np.abs(test_output[i] - normalize_result[i]))
    print(f"\nTest output: {test_output[i]},\nNetwork output: {normalize_result[i]},\nError: {test_error}")