import numpy as np
import matplotlib.pyplot as plt
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
        training_errors = []
        for epoch in range(epochs):
            # Ağı eğitme
            for i in range(len(inputs)):
                input_data = np.array([inputs[i]])
                target_data = np.array([targets[i]])

                # İleri ve geri yayılım
                output = self.forward(input_data)
                self.backward(input_data, target_data, 1 - (epoch/epochs))

                # Belirlenen epoch'ta bir hata kontrolü, training'de kullanılan 70 tanesinin hatası
                training_error = np.mean(np.abs(target_data - output))
                training_errors.append(training_error)
                if epoch % 10000 == 0:
                    print(f"Epoch {epoch}, Error: {training_error}\n")

        plt.plot(training_errors, label='Training Error', color='blue', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

# Model tanımı
input_size = 2
hidden_size1 = 4
hidden_size2 = 3
output_size = 1

nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

# Eğitim verileri
inputs_karma = np.array([[0.3,1.0],[0.3,0.2],[0.3,0.1],[0.1,0.3],[0.9,0.8],[0.9,0.2],[0.1,0.4],[1.0,0.3],[0.4,0.2],[0.2,0.2],[0.5,0.6],[0.7,0.6],[0.4,0.6],[0.9,0.6],[0.8,0.8],[1.0,0.7],[0.7,0.3],[1.0,0.1],[0.3,0.9],[1.0,0.6],[0.7,0.8],[0.2,1.0],[0.9,1.0],[0.1,0.8],[0.7,1.0],[0.8,0.7],[0.3,0.4],[0.7,0.4],[0.4,0.1],[0.5,0.3],[0.6,0.5],[0.3,0.5],[0.2,0.4],[1.0,0.9],[0.2,0.3],[0.2,0.9],[0.6,0.8],[0.5,0.9],[0.3,0.3],[0.7,0.2],[0.6,0.1],[0.6,0.3],[0.8,0.3],[0.4,0.3],[0.8,0.9],[0.8,0.4],[1.0,0.4],[0.3,0.7],[0.5,1.0],[0.2,0.8],[0.3,0.8],[0.6,0.9],[0.2,0.1],[0.8,0.6],[0.1,0.9],[0.1,1.0],[0.9,0.1],[0.8,0.5],[0.9,0.9],[0.1,0.7],[0.4,0.5],[0.7,0.5],[0.8,0.2],[0.5,0.8],[1.0,1.0],[0.2,0.5],[0.7,0.1],[0.9,0.4],[0.4,0.9],[0.1,0.6]])
targets_karma = np.array([[0.3],[0.06],[0.03],[0.03],[0.72],[0.18],[0.04],[0.3],[0.08],[0.04],[0.3],[0.42],[0.24],[0.54],[0.64],[0.7],[0.21],[0.1],[0.27],[0.6],[0.56],[0.2],[0.9],[0.08],[0.7],[0.56],[0.12],[0.28],[0.04],[0.15],[0.3],[0.15],[0.08],[0.9],[0.06],[0.18],[0.48],[0.45],[0.09],[0.14],[0.06],[0.18],[0.24],[0.12],[0.72],[0.32],[0.4],[0.21],[0.5],[0.16],[0.24],[0.54],[0.02],[0.48],[0.09],[0.1],[0.09],[0.4],[0.81],[0.07],[0.2],[0.35],[0.16],[0.4],[1.0],[0.1],[0.07],[0.36],[0.36],[0.06]])

inputs_odd = np.array([[0.3,0.1],[0.1,0.3],[0.7,0.3],[0.3,0.9],[0.5,0.3],[0.3,0.5],[0.5,0.9],[0.3,0.3],[0.3,0.7],[0.1,0.9],[0.9,0.1],[0.9,0.9],[0.1,0.7],[0.7,0.5],[0.7,0.1],[0.9,0.7],[0.5,0.7],[0.5,0.1],[0.9,0.3],[0.1,0.5],[0.9,0.5],[0.7,0.7],[0.7,0.9],[0.1,0.1],[0.5,0.5]])
targets_odd = np.array([[0.03],[0.03],[0.21],[0.27],[0.15],[0.15],[0.45],[0.09],[0.21],[0.09],[0.09],[0.81],[0.07],[0.35],[0.07],[0.63],[0.35],[0.05],[0.27],[0.05],[0.45],[0.49],[0.63],[0.01],[0.25]])

# Test verisi
test_input_karma = np.array([[0.9,0.7],[0.5,0.7],[0.5,0.1],[0.5,0.2],[0.4,0.8],[0.9,0.3],[0.6,0.4],[0.1,0.5],[1,0.5],[1,0.2],[1,0.8],[0.6,0.6],[0.3,0.6],[0.4,0.7],[0.6,0.7],[0.4,1],[0.9,0.5],[0.7,0.7],[0.9,0.9],[0.6,1],[0.6,0.2],[8,0.1],[0.1,0.2],[0.2,0.7],[0.1,0.1],[0.8,0.1],[0.5,0.5],[0.5,0.4],[0.2,0.6],[0.4,0.4]])
test_output_karma = np.array([63,35,5,10,32,27,24,5,50,20,80,36,18,28,42,40,45,49,81,60,12,80,2,14,1,8,25,20,12,16,])

test_input_even = np.array([[0.4,0.2],[0.2,0.2],[0.4,0.6],[0.8,0.8],[1.0,0.6],[0.2,1.0],[0.2,0.4],[0.6,0.8],[0.8,0.4],[1.0,0.4],[0.2,0.8],[0.8,0.6],[0.8,0.2],[1.0,1.0],[0.4,0.8],[0.6,0.4],[1.0,0.2],[1.0,0.8],[0.6,0.6],[0.4,1.0],[0.6,1.0],[0.6,0.2],[0.8,1.0],[0.2,0.6],[0.4,0.4]])
test_output_even = np.array([8,4,24,64,60,20,8,48,32,40,16,48,16,100,32,24,20,80,36,40,60,12,80,12,16,])

test_mode = input("Modeli hazir verilerle test etmek icin 1, kendiniz test verisi girmek icin 2'ye basin.\n")

if test_mode == "1": # otomatik test

    learning_settings = input("Modeli karma verilerle egitmek icin 1, sadece tek sayilarin carpimiyla egitip ciftlerle test icin 2'ye basin.\n")

    if learning_settings == "1": # karma veri eğitimi

        nn.train(inputs_karma, targets_karma, epochs= 50000)
        result = nn.forward(test_input_karma)
        normalize_result = result * 100

        for i in range(len(test_input_karma)):
            test_error = np.mean(np.abs(test_output_karma[i] - normalize_result[i]))
            print(f"\nTest output: {test_output_karma[i]},\nNetwork output: {normalize_result[i]},\nError: {test_error}")
    elif learning_settings == "2": # tek çarpımların eğitimi

        nn.train(inputs_odd, targets_odd, epochs= 50000)
        result = nn.forward(test_input_even)
        normalize_result = result * 100

        for i in range(len(test_input_even)):
            test_error = np.mean(np.abs(test_output_even[i] - normalize_result[i]))
            print(f"\nTest output: {test_output_even[i]},\nNetwork output: {normalize_result[i]},\nError: {test_error}")
elif test_mode == "2": # manual test

    learning_settings = input("Modeli karma verilerle egitmek icin 1, sadece tek sayilarin carpimiyla egitmek icin 2'ye basin.\n")
    
    if learning_settings == "1": #karma veri eğitimi

        x, y = map(int, input("Carpmak istediğiniz iki sayiyi girin (ornegin: 3 5):\n").split())

        test_input_manual = [[x/10,y/10]]
        test_output_manual = x * y

        nn.train(inputs_karma, targets_karma, epochs= 50000)
        result = nn.forward(test_input_manual)
        normalize_result = result * 100
        
        test_error = np.mean(np.abs(test_output_manual - normalize_result))
        print(f"\nTest output: {test_output_manual},\nNetwork output: {normalize_result},\nError: {test_error}")
    elif learning_settings == "2": # tek çarpımların eğitimi

        x, y = map(int, input("Carpmak istediğiniz iki sayiyi girin (ornegin: 3 5):\n").split())
        test_input_manual = [[x/10,y/10]]
        test_output_manual = x * y

        nn.train(inputs_odd, targets_odd, epochs= 50000)
        result = nn.forward(test_input_manual)
        normalize_result = result * 100
        
        test_error = np.mean(np.abs(test_output_manual - normalize_result))
        print(f"\nTest output: {test_output_manual},\nNetwork output: {normalize_result},\nError: {test_error}")