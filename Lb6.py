import random
import math

class CounterpropagationNetwork:
    def __init__(self, input_size, kohonen_nodes, output_size):
        self.input_size = input_size
        self.kohonen_nodes = kohonen_nodes
        self.output_size = output_size
        
        self.W_kohonen = [[random.uniform(0, 1) for _ in range(input_size)] for _ in range(kohonen_nodes)]
        
        self.W_grossberg = [[0.0 for _ in range(kohonen_nodes)] for _ in range(output_size)]

    def get_winner(self, x):
        min_dist = float('inf')
        winner_idx = -1
        for i in range(self.kohonen_nodes):
            dist = sum((x[j] - self.W_kohonen[i][j])**2 for j in range(self.input_size))
            if dist < min_dist:
                min_dist = dist
                winner_idx = i
        return winner_idx

    def train(self, X, Y, epochs=50, initial_alpha=0.6, initial_beta=0.6):
        for epoch in range(epochs):
            alpha = initial_alpha * (1 - epoch / epochs)
            beta = initial_beta * (1 - epoch / epochs)
            
            for x, y in zip(X, Y):
                winner = self.get_winner(x)
                
                for i in range(self.input_size):
                    self.W_kohonen[winner][i] += alpha * (x[i] - self.W_kohonen[winner][i])
                    
                for j in range(self.output_size):
                    self.W_grossberg[j][winner] += beta * (y[j] - self.W_grossberg[j][winner])

    def predict(self, x):
        winner = self.get_winner(x)
        outputs = [self.W_grossberg[j][winner] for j in range(self.output_size)]
        return outputs.index(max(outputs))

DIGITS = [
    "1111111111 1000000001 1000000001 1000000001 1000000001 1000000001 1000000001 1000000001 1111111111 0000000000", # 0
    "0000110000 0001110000 0000110000 0000110000 0000110000 0000110000 0000110000 0000110000 0011111100 0000000000", # 1
    "0011111100 0100000010 0000000010 0000000100 0000001000 0000010000 0000100000 0001000000 0111111110 0000000000", # 2
    "0011111100 0100000010 0000000010 0001111100 0000000010 0000000010 0100000010 0011111100 0000000000 0000000000", # 3
    "0001000100 0010000100 0100000100 0100000100 0111111100 0000000100 0000000100 0000000100 0000000000 0000000000"  # 4
]

def parse_digit(string_repr):
    return [int(char) for char in string_repr.replace(" ", "")]

def add_noise(pattern, noise_level=0.1):
    noisy = list(pattern)
    for i in range(len(noisy)):
        if random.random() < noise_level:
            noisy[i] = 1 - noisy[i] 
    return noisy

if __name__ == "__main__":
    random.seed(42)
    training_inputs = [parse_digit(d) for d in DIGITS]
    labels = [[1 if i == j else 0 for j in range(5)] for i in range(5)]

    cpn = CounterpropagationNetwork(input_size=100, kohonen_nodes=15, output_size=5)
    
    print("Обучение сети встречного распространения...")
    cpn.train(training_inputs, labels, epochs=30)
    
    print("\nТестирование на зашумленных данных (10% шума):")
    for i, ideal_pattern in enumerate(training_inputs):
        noisy = add_noise(ideal_pattern, 0.10)
        prediction = cpn.predict(noisy)
        print(f"Подана цифра {i} (с шумом) -> Распознана как: {prediction}")
