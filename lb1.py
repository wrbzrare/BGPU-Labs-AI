class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=20):
        self.weights = [0.0] * input_size
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        weighted_sum = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        return 1 if weighted_sum > 0 else 0

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                
                if error != 0:
                    for i in range(len(self.weights)):
                        self.weights[i] += self.learning_rate * error * inputs[i]
                    self.bias += self.learning_rate * error


if __name__ == "__main__":
    print("=== Обучение функции И (AND) ===")
    and_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    and_labels = [0, 0, 0, 1]
    
    perceptron_and = Perceptron(input_size=2)
    perceptron_and.train(and_inputs, and_labels)
    
    for x in and_inputs:
        print(f"Вход {x} -> Выход: {perceptron_and.predict(x)}")

    print("\n=== Обучение функции ИЛИ (OR) ===")
    or_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    or_labels = [0, 1, 1, 1]
    
    perceptron_or = Perceptron(input_size=2)
    perceptron_or.train(or_inputs, or_labels)
    
    for x in or_inputs:
        print(f"Вход {x} -> Выход: {perceptron_or.predict(x)}")

    print("\n=== Обучение функции НЕ (NOT) ===")
    not_inputs = [[0], [1]]
    not_labels = [1, 0]
    
    perceptron_not = Perceptron(input_size=1)
    perceptron_not.train(not_inputs, not_labels)
    
    for x in not_inputs:
        print(f"Вход {x} -> Выход: {perceptron_not.predict(x)}")