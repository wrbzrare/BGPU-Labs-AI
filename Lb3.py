'''
Обучить многослойный перцептрон логической функции "исключающее или". Убедиться, что такая архитектура сохраняет способность обучаться простым логическим функциям.
'''
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.lr = learning_rate
        
        self.W1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        
        self.W2 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [random.uniform(-1, 1) for _ in range(output_size)]

    def predict(self, x):
        self.hidden_outputs = []
        for j in range(len(self.b1)):
            activation = self.b1[j]
            for i in range(len(x)):
                activation += x[i] * self.W1[i][j]
            self.hidden_outputs.append(sigmoid(activation))
            
        self.final_outputs = []
        for k in range(len(self.b2)):
            activation = self.b2[k]
            for j in range(len(self.hidden_outputs)):
                activation += self.hidden_outputs[j] * self.W2[j][k]
            self.final_outputs.append(sigmoid(activation))
            
        return self.final_outputs

    def train_step(self, x, target):
        self.predict(x)
        
        output_deltas = []
        for k in range(len(self.b2)):
            error = target[k] - self.final_outputs[k]
            delta = error * sigmoid_derivative(self.final_outputs[k])
            output_deltas.append(delta)
            
        hidden_deltas = []
        for j in range(len(self.b1)):
            error = 0.0
            for k in range(len(self.b2)):
                error += output_deltas[k] * self.W2[j][k]
            delta = error * sigmoid_derivative(self.hidden_outputs[j])
            hidden_deltas.append(delta)
            
        for j in range(len(self.hidden_outputs)):
            for k in range(len(self.b2)):
                self.W2[j][k] += self.lr * output_deltas[k] * self.hidden_outputs[j]
        for k in range(len(self.b2)):
            self.b2[k] += self.lr * output_deltas[k]
            
        # Обновляем веса W1 и b1 (от входа к скрытому)
        for i in range(len(x)):
            for j in range(len(self.b1)):
                self.W1[i][j] += self.lr * hidden_deltas[j] * x[i]
        for j in range(len(self.b1)):
            self.b1[j] += self.lr * hidden_deltas[j]

    def fit(self, training_data, labels, epochs=10000):
        for _ in range(epochs):
            # В каждой эпохе проходим по всем обучающим примерам
            for x, y in zip(training_data, labels):
                self.train_step(x, [y])


def test_network(name, inputs, labels, epochs=10000):
    print(f"\n=== Обучение функции {name} ===")
    nn = MultiLayerPerceptron(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)
    nn.fit(inputs, labels, epochs=epochs)
    
    for x in inputs:
        raw_output = nn.predict(x)[0]
        predicted_class = 1 if raw_output >= 0.5 else 0
        print(f"Вход {x} -> Выход сети: {raw_output:.4f} (Округленно: {predicted_class})")

if __name__ == "__main__":
    random.seed(42)

    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_labels = [0, 1, 1, 0]
    test_network("XOR (Исключающее ИЛИ)", xor_inputs, xor_labels, epochs=15000)

    and_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    and_labels = [0, 0, 0, 1]
    test_network("AND (И)", and_inputs, and_labels, epochs=5000)

    or_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    or_labels = [0, 1, 1, 1]
    test_network("OR (ИЛИ)", or_inputs, or_labels, epochs=5000)
