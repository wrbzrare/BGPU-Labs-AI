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
            indices = list(range(self.size))
            random.shuffle(indices)
            
            for i in indices:
                net_input = sum(self.weights[i][j] * state[j] for j in range(self.size))
                state[i] = 1 if net_input >= 0 else -1
                
        return state


DIGITS = {
    "0": """
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
    """,
    "1": """
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
    """,
    "4": """
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
    """,
    "7": """
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
    """
}

def parse_digit(ascii_str):
    lines = [line.strip() for line in ascii_str.strip().split('\n')]
    flat_list = []
    for line in lines:
        for char in line:
            flat_list.append(1 if char == '#' else -1)
    return flat_list

def print_digit(pattern, width=10):
    for i in range(0, len(pattern), width):
        row = pattern[i:i+width]
        print("".join(['#' if x == 1 else '.' for x in row]))

def add_noise(pattern, noise_level=0.3):
    noisy = list(pattern)
    for i in range(len(noisy)):
        if random.random() < noise_level:
            noisy[i] = -noisy[i]
    return noisy

if __name__ == "__main__":
    random.seed(42)

    patterns = [parse_digit(DIGITS[key]) for key in DIGITS]
    
    print("Обучение сети на цифрах: 0, 1, 4, 7...")
    hopfield = HopfieldNetwork(size=100)
    hopfield.train(patterns)
    
    ideal_pattern = parse_digit(DIGITS["4"])
    noisy_pattern = add_noise(ideal_pattern, noise_level=0.3)
    
    print("\n=== Зашумленный образ (30% помех) ===")
    print_digit(noisy_pattern)
    
    restored_pattern = hopfield.restore(noisy_pattern, steps=5)
    
    print("\n=== Восстановленный образ ===")
    print_digit(restored_pattern)
    
    errors = sum(1 for i, j in zip(ideal_pattern, restored_pattern) if i != j)
    print(f"\nОшибок после восстановления: {errors} из 100 пикселей.")
