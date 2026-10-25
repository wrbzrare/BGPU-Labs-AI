'''
Обучить однослойный перцептрон из 100 входов (10х10) и 10 выходов (0-9) распознаванию арабских цифр
'''
class MultiClassPerceptron:
    def __init__(self, input_size=100, num_classes=10, learning_rate=0.1, epochs=50):
        # weights — это матрица 10 (классов) х 100 (входов)
        # biases — массив из 10 смещений
        self.weights = [[0.0] * input_size for _ in range(num_classes)]
        self.biases = [0.0] * num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_classes = num_classes

    def predict_raw(self, inputs):
        """Возвращает сырые значения взвешенной суммы для всех 10 нейронов."""
        scores = []
        for class_idx in range(self.num_classes):
            w_sum = sum(x * w for x, w in zip(inputs, self.weights[class_idx])) + self.biases[class_idx]
            scores.append(w_sum)
        return scores

    def predict_outputs(self, inputs):
        """Возвращает массив из 10 выходов (0 или 1) после функции активации."""
        scores = self.predict_raw(inputs)
        return [1 if score > 0 else 0 for score in scores]

    def predict_digit(self, inputs):
        """Возвращает класс (0-9) с максимальным уровнем сигнала."""
        scores = self.predict_raw(inputs)
        return scores.index(max(scores))

    def train(self, training_data, labels):
        for epoch in range(self.epochs):
            total_errors = 0
            for inputs, label in zip(training_data, labels):
                # Формируем целевой вектор, где 1 стоит только на позиции правильной цифры
                targets = [1 if i == label else 0 for i in range(self.num_classes)]
                outputs = self.predict_outputs(inputs)
                
                # Обновляем веса каждого из 10 нейронов независимо
                for class_idx in range(self.num_classes):
                    error = targets[class_idx] - outputs[class_idx]
                    
                    if error != 0:
                        total_errors += 1
                        for i in range(len(inputs)):
                            self.weights[class_idx][i] += self.learning_rate * error * inputs[i]
                        self.biases[class_idx] += self.learning_rate * error
            
            # Ранняя остановка, если сеть больше не ошибается на обучающей выборке
            if total_errors == 0:
                print(f"Сеть сошлась на {epoch+1}-й эпохе.")
                break


# --- Подготовка данных (Датасет 10x10) ---

# Символ '#' = 1 (закрашенный пиксель), '.' = 0 (пустой)
DIGITS_ASCII = [
"""
..######..
.#......#.
.#......#.
.#......#.
.#......#.
.#......#.
.#......#.
.#......#.
..######..
..........
""", # 0
"""
....##....
...###....
....##....
....##....
....##....
....##....
....##....
....##....
..######..
..........
""", # 1
"""
..######..
.#......#.
........#.
.......#..
......#...
.....#....
....#.....
...#......
.########.
..........
""", # 2
"""
..######..
.#......#.
........#.
...#####..
........#.
........#.
.#......#.
..######..
..........
..........
""", # 3
"""
...#...#..
..#....#..
.#.....#..
.#.....#..
.#######..
.......#..
.......#..
.......#..
..........
..........
""", # 4
"""
.#######..
.#........
.#........
.######...
.......#..
.......#..
.#.....#..
..#####...
..........
..........
""", # 5
"""
...####...
..#.......
.#........
.######...
.#.....#..
.#.....#..
.#.....#..
..#####...
..........
..........
""", # 6
"""
.########.
........#.
.......#..
......#...
.....#....
....#.....
...#......
..#.......
..........
..........
""", # 7
"""
..#####...
.#.....#..
.#.....#..
..#####...
.#.....#..
.#.....#..
.#.....#..
..#####...
..........
..........
""", # 8
"""
..#####...
.#.....#..
.#.....#..
..######..
.......#..
.......#..
......#...
...###....
..........
..........
"""  # 9
]

def parse_digit(ascii_str):
    """Преобразует строку 10x10 в одномерный массив из 100 нулей и единиц."""
    lines = [line.strip() for line in ascii_str.strip().split('\n')]
    flat_list = []
    for line in lines:
        for char in line:
            flat_list.append(1 if char == '#' else 0)
    return flat_list

if __name__ == "__main__":
    # Подготовка обучающей выборки
    training_inputs = [parse_digit(d) for d in DIGITS_ASCII]
    labels = list(range(10)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Инициализация и обучение
    print("Начало обучения на матрицах 10x10...")
    nn = MultiClassPerceptron(input_size=100, num_classes=10)
    nn.train(training_inputs, labels)

    # Тестирование на идеальных данных
    print("\nПроверка на обучающей выборке:")
    for i, x in enumerate(training_inputs):
        prediction = nn.predict_digit(x)
        print(f"Подана цифра {i} -> Распознана как: {prediction}")

    # Тестирование на зашумленных данных (поврежденная единица)
    noisy_1 = """
    ....##....
    ...#.#....
    ....##....
    ....#.....
    ....##....
    ..........
    ....##....
    ....##....
    ..###.##..
    ..........
    """
    print("\nПроверка на цифре '1' с помехами и пропусками пикселей:")
    noisy_input = parse_digit(noisy_1)
    print(f"Распознана как: {nn.predict_digit(noisy_input)}")
