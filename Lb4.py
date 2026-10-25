import random

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = [[0.0] * size for _ in range(size)]

    def train(self, patterns):
        for p in patterns:
            for i in range(self.size):
                for j in range(self.size):
                    if i != j:  
                        self.weights[i][j] += p[i] * p[j]
                        
        for i in range(self.size):
            for j in range(self.size):
                self.weights[i][j] /= self.size

    def restore(self, pattern, steps=5):
        state = list(pattern)
        for _ in range(steps):
            for i in range(self.size):
                net_input = sum(self.weights[i][j] * state[j] for j in range(self.size))
                state[i] = 1 if net_input >= 0 else -1
        return state

class PerceptronClassifier:
    def __init__(self, input_size, num_classes, learning_rate=0.1):
        self.weights = [[0.0] * input_size for _ in range(num_classes)]
        self.biases = [0.0] * num_classes
        self.lr = learning_rate
        self.num_classes = num_classes

    def train(self, training_data, labels, epochs=50):
        for _ in range(epochs):
            for inputs, label in zip(training_data, labels):
                targets = [1 if i == label else 0 for i in range(self.num_classes)]
                
                outputs = []
                for class_idx in range(self.num_classes):
                    w_sum = sum(x * w for x, w in zip(inputs, self.weights[class_idx])) + self.biases[class_idx]
                    outputs.append(1 if w_sum > 0 else 0)
                    
                for class_idx in range(self.num_classes):
                    error = targets[class_idx] - outputs[class_idx]
                    if error != 0:
                        for i in range(len(inputs)):
                            self.weights[class_idx][i] += self.lr * error * inputs[i]
                        self.biases[class_idx] += self.lr * error

    def predict(self, inputs):
        scores = []
        for class_idx in range(self.num_classes):
            w_sum = sum(x * w for x, w in zip(inputs, self.weights[class_idx])) + self.biases[class_idx]
            scores.append(w_sum)
        return scores.index(max(scores))

def parse_pattern(ascii_str):
    # Преобразуем '#' в 1, а '.' в -1
    return [1 if char == '#' else -1 for char in ascii_str if char in ['.', '#']]

def print_pattern(pattern, width=5):
    for i in range(0, len(pattern), width):
        row = pattern[i:i+width]
        print(" ".join(['#' if x == 1 else '.' for x in row]))

def add_noise(pattern, noise_level=0.2):
    noisy = list(pattern)
    for i in range(len(noisy)):
        if random.random() < noise_level:
            noisy[i] = -noisy[i]  
    return noisy

if __name__ == "__main__":
    shape_T = """
    # # # # #
    . . # . .
    . . # . .
    . . # . .
    . . # . .
    """
    
    shape_X = """
    # . . . #
    . # . # .
    . . # . .
    . # . # .
    # . . . #
    """
    
    shape_O = """
    . # # # .
    # . . . #
    # . . . #
    # . . . #
    . # # # .
    """
    
    patterns_str = [shape_T, shape_X, shape_O]
    class_names = ["Буква T", "Буква X", "Буква O"]
    
    training_patterns = [parse_pattern(p) for p in patterns_str]
    labels = [0, 1, 2]

    print("Обучение сети Хопфилда (запоминание образов)...")
    hopfield = HopfieldNetwork(size=25)
    hopfield.train(training_patterns)

    print("Обучение Перцептрона (классификация)...")
    classifier = PerceptronClassifier(input_size=25, num_classes=3)
    classifier.train(training_patterns, labels)

    print("\n--- ТЕСТИРОВАНИЕ ---")
    
    ideal_pattern = training_patterns[1] 
    noisy_pattern = add_noise(ideal_pattern, noise_level=0.20)
    
    print("\n1. Зашумленный образ (Подан на вход сети Хопфилда):")
    print_pattern(noisy_pattern)
    
    pred_noisy = classifier.predict(noisy_pattern)
    print(f"Перцептрон видит в зашумленном образе: {class_names[pred_noisy]}")
    
    restored_pattern = hopfield.restore(noisy_pattern)
    
    print("\n2. Восстановленный образ (Выход сети Хопфилда / Вход перцептрона):")
    print_pattern(restored_pattern)
    
    pred_restored = classifier.predict(restored_pattern)
    print(f"Перцептрон уверенно распознает восстановленный образ: {class_names[pred_restored]}")
