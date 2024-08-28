import numpy as np
import random
from itertools import combinations

cities = [
    (1, 37, 52), (2, 49, 49), (3, 52, 64), (4, 20, 26), (5, 40, 30), (6, 21, 47),
    (7, 17, 63), (8, 31, 62), (9, 52, 33), (10, 51, 21), (11, 42, 41), (12, 31, 32),
    (13, 5, 25), (14, 12, 42), (15, 36, 16), (16, 52, 41), (17, 27, 23), (18, 17, 33),
    (19, 13, 13), (20, 57, 58), (21, 62, 42), (22, 42, 57), (23, 16, 57), (24, 8, 52),
    (25, 7, 38), (26, 27, 68), (27, 30, 48), (28, 43, 67), (29, 58, 48), (30, 58, 27),
    (31, 37, 69), (32, 38, 46), (33, 46, 10), (34, 61, 33), (35, 62, 63), (36, 63, 69),
    (37, 32, 22), (38, 45, 35), (39, 59, 15), (40, 5, 6), (41, 10, 17), (42, 21, 10),
    (43, 5, 64), (44, 30, 15), (45, 39, 10), (46, 32, 39), (47, 25, 32), (48, 25, 55),
    (49, 48, 28), (50, 56, 37), (51, 30, 40)
]

city_coords = np.array([(city[1], city[2]) for city in cities])


def euclidean_distance(city1, city2):
    return np.linalg.norm(city1 - city2)


def calculate_total_distance(route, city_coords):
    total_distance = 0
    for i in range(len(route)):
        total_distance += euclidean_distance(city_coords[route[i - 1]], city_coords[route[i]])
    return total_distance


def generate_initial_population(pop_size, city_count):
    population = []
    for _ in range(pop_size):
        route = np.random.permutation(city_count)
        population.append(route)
    return population


def evaluate_population(population, city_coords):
    fitness_scores = []
    for route in population:
        total_distance = calculate_total_distance(route, city_coords)
        fitness = 1 / total_distance
        fitness_scores.append(fitness)
    return fitness_scores


def select_parents(population, fitness_scores, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        tournament_winner = max(tournament, key=lambda x: x[1])[0]
        selected_parents.append(tournament_winner)
    return selected_parents


def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end + 1] = parent1[start:end + 1]
    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = gene
    return child


def swap_mutation(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            swap_with = random.randint(0, len(route) - 1)
            route[i], route[swap_with] = route[swap_with], route[i]
    return route


def generate_next_generation(population, fitness_scores, city_coords, mutation_rate, tournament_size):
    new_population = []
    parents = select_parents(population, fitness_scores, tournament_size)
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[(i + 1) % len(parents)]
        child1 = order_crossover(parent1, parent2)
        child2 = order_crossover(parent2, parent1)
        new_population.append(swap_mutation(child1, mutation_rate))
        new_population.append(swap_mutation(child2, mutation_rate))
    return new_population


def genetic_algorithm_tsp(city_coords, pop_size=100, generations=500, mutation_rate=0.02, tournament_size=5):
    city_count = len(city_coords)
    population = generate_initial_population(pop_size, city_count)
    best_route = None
    best_distance = float('inf')

    for generation in range(generations):
        fitness_scores = evaluate_population(population, city_coords)
        best_index = np.argmax(fitness_scores)
        best_route_in_gen = population[best_index]
        best_distance_in_gen = calculate_total_distance(best_route_in_gen, city_coords)

        if best_distance_in_gen < best_distance:
            best_route = best_route_in_gen
            best_distance = best_distance_in_gen

        population = generate_next_generation(population, fitness_scores, city_coords, mutation_rate, tournament_size)
        print(f"Generation {generation + 1}: Best Distance = {best_distance:.2f}")

    return best_route, best_distance


best_route, best_distance = genetic_algorithm_tsp(city_coords)
print("Best Route: ", best_route)
print("Best Distance: ", best_distance)
