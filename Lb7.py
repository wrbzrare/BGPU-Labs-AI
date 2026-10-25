'''
Реализовать и обучить когнитрон распознаванию произвольных образов.
'''
import math
import random

class CognitronLayer:
    def __init__(self, input_size, num_neurons, learning_rate=0.1):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.lr = learning_rate
        
        self.weights = [[random.uniform(0.1, 1.0) for _ in range(input_size)] for _ in range(num_neurons)]
        self._normalize_weights()

    def _normalize_weights(self):
        for i in range(self.num_neurons):
            norm = math.sqrt(sum(w**2 for w in self.weights[i]))
            if norm > 0:
                self.weights[i] = [w / norm for w in self.weights[i]]

    def activate(self, x):
        activations = []
        for i in range(self.num_neurons):
            activation = sum(x[j] * self.weights[i][j] for j in range(self.input_size))
            activations.append(activation)
            
        return activations

    def get_winner(self, activations):
        return activations.index(max(activations))

    def train(self, patterns, epochs=20):
        for epoch in range(epochs):
            for x in patterns:
                activations = self.activate(x)
                
                winner_idx = self.get_winner(activations)
                
                for j in range(self.input_size):
                    self.weights[winner_idx][j] += self.lr * (x[j] - self.weights[winner_idx][j])
                    
            self._normalize_weights()

    def recognize(self, x):
        activations = self.activate(x)
        return self.get_winner(activations)


if __name__ == "__main__":
    random.seed(42)

    # Образ "Вертикальная линия"
    pattern_V = [
        0, 1, 0,
        0, 1, 0,
        0, 1, 0
    ]
    
    # Образ "Горизонтальная линия"
    pattern_H = [
        0, 0, 0,
        1, 1, 1,
        0, 0, 0
    ]
    
    # Образ "Диагональ"
    pattern_D = [
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    ]

    training_data = [pattern_V, pattern_H, pattern_D]

    print("Инициализация и обучение когнитрона...")
    cognitron = CognitronLayer(input_size=9, num_neurons=3, learning_rate=0.2)
    cognitron.train(training_data, epochs=50)

    print("\nПроверка реакции нейронов на знакомые образы:")
    neuron_V = cognitron.recognize(pattern_V)
    neuron_H = cognitron.recognize(pattern_H)
    neuron_D = cognitron.recognize(pattern_D)
    
    print(f"Вертикальная линия активировала нейрон: {neuron_V}")
    print(f"Горизонтальная линия активировала нейрон: {neuron_H}")
    print(f"Диагональ активировала нейрон: {neuron_D}")

    noisy_V = [
        0, 1, 0,
        1, 1, 0, # Помеха слева
        0, 1, 0
    ]
    
    predicted_noisy = cognitron.recognize(noisy_V)
    print(f"\nЗашумленная вертикальная линия активировала нейрон: {predicted_noisy}")
    if predicted_noisy == neuron_V:
        print("Сеть успешно обобщила образ несмотря на помехи.")
