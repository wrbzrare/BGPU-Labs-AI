'''
С помощью генетического алгоритма найти x, y для функции f(x,y)=1/(1+x^2+y^2)
Сравнить производительность отбора методом элит и методом рулетки.
'''

import random
import time

def fitness(x, y):
    return 1.0 / (1.0 + x**2 + y**2)

def init_population(size, bounds):
    return [[random.uniform(bounds[0], bounds[1]), 
             random.uniform(bounds[0], bounds[1])] for _ in range(size)]

def selection_roulette(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind, fit in zip(population, fitnesses):
        current += fit
        if current > pick:
            return ind
    return population[-1]

def selection_elite(population, fitnesses, elite_fraction=0.2):
    sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda p: p[0], reverse=True)]
    elite_count = max(1, int(len(population) * elite_fraction))
    return random.choice(sorted_pop[:elite_count])

def crossover(p1, p2):
    alpha = random.random()
    c1 = [alpha * p1[0] + (1 - alpha) * p2[0], alpha * p1[1] + (1 - alpha) * p2[1]]
    c2 = [(1 - alpha) * p1[0] + alpha * p2[0], (1 - alpha) * p1[1] + alpha * p2[1]]
    return c1, c2

def mutate(ind, bounds, mutation_rate=0.1):
    for i in range(2):
        if random.random() < mutation_rate:
            ind[i] += random.gauss(0, 0.5)
            # Ограничиваем выход за пределы поиска
            ind[i] = max(bounds[0], min(bounds[1], ind[i]))
    return ind

def run_ga(selection_method, pop_size=100, generations=50, bounds=[-5, 5]):
    population = init_population(pop_size, bounds)
    
    start_time = time.perf_counter()
    
    for _ in range(generations):
        fitnesses = [fitness(ind[0], ind[1]) for ind in population]
        new_population = []
        
        best_idx = fitnesses.index(max(fitnesses))
        new_population.append(population[best_idx][:])
        
        while len(new_population) < pop_size:
            if selection_method == 'roulette':
                p1 = selection_roulette(population, fitnesses)
                p2 = selection_roulette(population, fitnesses)
            else:
                p1 = selection_elite(population, fitnesses)
                p2 = selection_elite(population, fitnesses)
                
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1, bounds))
            if len(new_population) < pop_size:
                new_population.append(mutate(c2, bounds))
                
        population = new_population

    end_time = time.perf_counter()
    
    final_fitnesses = [fitness(ind[0], ind[1]) for ind in population]
    best_idx = final_fitnesses.index(max(final_fitnesses))
    best_ind = population[best_idx]
    
    return best_ind, max(final_fitnesses), end_time - start_time


if __name__ == "__main__":
    random.seed(42)
    
    print("=== Метод рулетки ===")
    best_r, fit_r, time_r = run_ga('roulette', pop_size=50, generations=50)
    print(f"Лучшая точка: x = {best_r[0]:.4f}, y = {best_r[1]:.4f}")
    print(f"Значение функции: {fit_r:.6f} (Идеал: 1.0)")
    print(f"Время выполнения: {time_r:.4f} сек\n")
    
    print("=== Метод элит (Турнир/Усечение) ===")
    best_e, fit_e, time_e = run_ga('elite', pop_size=50, generations=50)
    print(f"Лучшая точка: x = {best_e[0]:.4f}, y = {best_e[1]:.4f}")
    print(f"Значение функции: {fit_e:.6f} (Идеал: 1.0)")
    print(f"Время выполнения: {time_e:.4f} сек")
