'''
Обучить перцептрон (однослойный, затем многослойный) при помощи генетического алгоритма.
'''

import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-max(min(x, 20), -20)))

def calculate_fitness(network, chromosome, inputs, targets):
    error = 0.0
    for x, target in zip(inputs, targets):
        prediction = network.predict(x, chromosome)
        error += (prediction - target) ** 2
    return 1.0 / (1.0 + error)

class SingleLayerPerceptron:
    def __init__(self):
        self.num_genes = 3
        
    def predict(self, x, chromosome):
        w1, w2, bias = chromosome[0], chromosome[1], chromosome[2]
        return sigmoid(x[0]*w1 + x[1]*w2 + bias)

class MultiLayerPerceptron:
    def __init__(self):
        self.num_genes = 9
        
    def predict(self, x, chromosome):
        W1_00, W1_01, W1_10, W1_11 = chromosome[0:4]
        b1_0, b1_1 = chromosome[4:6]
        W2_0, W2_1 = chromosome[6:8]
        b2 = chromosome[8]
        
        h1 = sigmoid(x[0]*W1_00 + x[1]*W1_01 + b1_0)
        h2 = sigmoid(x[0]*W1_10 + x[1]*W1_11 + b1_1)
        
        return sigmoid(h1*W2_0 + h2*W2_1 + b2)


def run_ga(network, inputs, targets, pop_size=100, generations=300):
    population = [[random.uniform(-5, 5) for _ in range(network.num_genes)] 
                  for _ in range(pop_size)]
    
    for gen in range(generations):
        fitnesses = [calculate_fitness(network, chrom, inputs, targets) for chrom in population]
        
        best_idx = fitnesses.index(max(fitnesses))
        best_chrom = population[best_idx]
        
        if max(fitnesses) > 0.99:
            print(f"  [!] Алгоритм сошелся на {gen}-м поколении.")
            break
            
        new_population = [best_chrom[:]] # Элитизм: сохраняем лучшего без изменений
        
        while len(new_population) < pop_size:
            parent1 = max(random.sample(list(zip(population, fitnesses)), 3), key=lambda item: item[1])[0]
            parent2 = max(random.sample(list(zip(population, fitnesses)), 3), key=lambda item: item[1])[0]
          
            child = []
            for g1, g2 in zip(parent1, parent2):
                child.append(g1 if random.random() < 0.5 else g2)
                
            for i in range(len(child)):
                if random.random() < 0.1: # Шанс мутации гена 10%
                    child[i] += random.gauss(0, 1.0)
                    
            new_population.append(child)
            
        population = new_population
        
    return max(population, key=lambda chrom: calculate_fitness(network, chrom, inputs, targets))

if __name__ == "__main__":
    random.seed(42)

    print("=== 1. Обучение ОДНОСЛОЙНОГО перцептрона функции AND ===")
    and_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    and_targets = [0, 0, 0, 1]
    
    single_net = SingleLayerPerceptron()
    best_single = run_ga(single_net, and_inputs, and_targets, pop_size=50, generations=100)
    
    print("Результаты (AND):")
    for x in and_inputs:
        pred = single_net.predict(x, best_single)
        print(f"Вход {x} -> Выход: {pred:.4f} (Округленно: {round(pred)})")


    print("\n=== 2. Обучение МНОГОСЛОЙНОГО перцептрона функции XOR ===")
    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_targets = [0, 1, 1, 0]
    
    multi_net = MultiLayerPerceptron()
    best_multi = run_ga(multi_net, xor_inputs, xor_targets, pop_size=150, generations=500)
    
    print("Результаты (XOR):")
    for x in xor_inputs:
        pred = multi_net.predict(x, best_multi)
        print(f"Вход {x} -> Выход: {pred:.4f} (Округленно: {round(pred)})")
