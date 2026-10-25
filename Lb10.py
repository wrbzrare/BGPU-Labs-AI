'''
Реализовать модель биоценоза растения-травоядные-хищники.
Снабдить агентов однослойными перцептронами в качестве контроллеров их активности в среде.
Продумать модель метаболизма. Реализовать графический интерфейс приложения.
'''

import tkinter as tk
import math
import random

WIDTH, HEIGHT = 800, 600
NUM_PLANTS = 40
NUM_HERBIVORES = 20
NUM_PREDATORS = 5

class Perceptron:
    def __init__(self, weights=None):
        self.input_size = 5
        self.output_size = 2 
        
        if weights:
            self.weights = weights
        else:
            self.weights = [[random.uniform(-1, 1) for _ in range(self.input_size)] 
                            for _ in range(self.output_size)]

    def predict(self, inputs):
        outputs = []
        for i in range(self.output_size):
            activation = sum(inputs[j] * self.weights[i][j] for j in range(self.input_size))
            outputs.append(math.tanh(activation))
        return outputs

    def mutate(self, mutation_rate=0.1):
        new_weights = []
        for row in self.weights:
            new_row = [w + random.gauss(0, 0.2) if random.random() < mutation_rate else w for w in row]
            new_weights.append(new_row)
        return Perceptron(new_weights)


class Agent:
    def __init__(self, x, y, agent_type, brain=None):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * math.pi)
        self.type = agent_type
        self.energy = 150.0
        self.brain = brain if brain else Perceptron()
        self.radius = 5 if agent_type == 'herbivore' else 7
        self.speed = 0
        self.max_speed = 3.0 if agent_type == 'herbivore' else 3.5

    def get_closest(self, targets):
        closest = None
        min_dist = float('inf')
        for t in targets:
            dist = math.hypot(self.x - t.x, self.y - t.y)
            if dist < min_dist and dist > 0:
                min_dist = dist
                closest = t
        return closest, min_dist

    def update(self, plants, herbivores, predators):
        food_targets = plants if self.type == 'herbivore' else herbivores
        threat_targets = predators if self.type == 'herbivore' else []

        closest_food, dist_food = self.get_closest(food_targets)
        closest_threat, dist_threat = self.get_closest(threat_targets)

        norm_dist_food = 1.0 - min(dist_food / 300.0, 1.0) if closest_food else 0.0
        norm_dist_threat = 1.0 - min(dist_threat / 300.0, 1.0) if closest_threat else 0.0

        angle_food = math.atan2(closest_food.y - self.y, closest_food.x - self.x) - self.angle if closest_food else 0.0
        angle_threat = math.atan2(closest_threat.y - self.y, closest_threat.x - self.x) - self.angle if closest_threat else 0.0

        angle_food = (angle_food + math.pi) % (2 * math.pi) - math.pi
        angle_threat = (angle_threat + math.pi) % (2 * math.pi) - math.pi

        inputs = [norm_dist_food, angle_food, norm_dist_threat, angle_threat, 1.0]

        acceleration, turn = self.brain.predict(inputs)

        self.angle += turn * 0.2
        self.speed = max(0, min(self.max_speed, self.speed + acceleration))

        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        self.x %= WIDTH
        self.y %= HEIGHT

        base_metabolism = 0.1
        kinetic_metabolism = (self.speed ** 2) * 0.02
        self.energy -= (base_metabolism + kinetic_metabolism)


class Plant:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = 20.0
        self.type_name = random.choice(["Бамбук", "Тростник"])

    def grow(self, is_day):
        if not is_day and self.energy < 100:
            self.energy += 0.5


class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Биоценоз: Перцептроны и Метаболизм")
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
        self.canvas.pack()

        self.time_ticks = 0
        self.is_day = True

        self.plants = [Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NUM_PLANTS)]
        self.herbivores = [Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT), 'herbivore') for _ in range(NUM_HERBIVORES)]
        self.predators = [Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT), 'predator') for _ in range(NUM_PREDATORS)]

        self.update_simulation()

    def handle_interactions(self):
        for h in self.herbivores:
            for p in self.plants[:]:
                if math.hypot(h.x - p.x, h.y - p.y) < h.radius + 5:
                    h.energy += p.energy
                    self.plants.remove(p)
                    self.plants.append(Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)))

        for p in self.predators:
            for h in self.herbivores[:]:
                if math.hypot(p.x - h.x, p.y - h.y) < p.radius + h.radius:
                    p.energy += h.energy
                    self.herbivores.remove(h)

    def process_lifecycle(self):
        self.herbivores = [h for h in self.herbivores if h.energy > 0]
        self.predators = [p for p in self.predators if p.energy > 0]

        new_herbivores = []
        for h in self.herbivores:
            if h.energy > 250:
                h.energy /= 2
                child = Agent(h.x, h.y, 'herbivore', h.brain.mutate())
                child.energy = h.energy
                new_herbivores.append(child)
        self.herbivores.extend(new_herbivores)

        new_predators = []
        for p in self.predators:
            if p.energy > 300:
                p.energy /= 2
                child = Agent(p.x, p.y, 'predator', p.brain.mutate())
                child.energy = p.energy
                new_predators.append(child)
        self.predators.extend(new_predators)

    def draw(self):
        self.canvas.delete("all")
        
        bg_color = "#e6f7ff" if self.is_day else "#1a252c"
        self.canvas.configure(bg=bg_color)
        
        cycle_text = "ДЕНЬ (Фотосинтез)" if self.is_day else "НОЧЬ (Фаза роста растений)"
        self.canvas.create_text(10, 10, anchor="nw", text=f"Цикл: {cycle_text}\nТравоядных: {len(self.herbivores)}\nХищников: {len(self.predators)}", fill="gray")

        for p in self.plants:
            r = min(3 + (p.energy / 20), 8)
            color = "#228B22" if self.is_day else "#145214"
            self.canvas.create_oval(p.x - r, p.y - r, p.x + r, p.y + r, fill=color, outline="")

        for h in self.herbivores:
            self.canvas.create_oval(h.x - h.radius, h.y - h.radius, h.x + h.radius, h.y + h.radius, fill="blue")
            # Направление взгляда
            self.canvas.create_line(h.x, h.y, h.x + math.cos(h.angle)*10, h.y + math.sin(h.angle)*10, fill="cyan")

        for p in self.predators:
            self.canvas.create_oval(p.x - p.radius, p.y - p.radius, p.x + p.radius, p.y + p.radius, fill="red")
            self.canvas.create_line(p.x, p.y, p.x + math.cos(p.angle)*15, p.y + math.sin(p.angle)*15, fill="yellow")

    def update_simulation(self):
        self.time_ticks += 1
        
        if self.time_ticks % 200 == 0:
            self.is_day = not self.is_day

        for p in self.plants:
            p.grow(self.is_day)

        for h in self.herbivores:
            h.update(self.plants, self.herbivores, self.predators)
            
        for p in self.predators:
            p.update(self.plants, self.herbivores, self.predators)

        self.handle_interactions()
        self.process_lifecycle()
        self.draw()

        if len(self.herbivores) == 0:
            self.herbivores = [Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT), 'herbivore') for _ in range(5)]
        
        self.root.after(30, self.update_simulation)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()
