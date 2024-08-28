import random
from deap import base, creator, tools, algorithms

# Define the fitness function
def fitness_function(individual):
    x = individual[0]
    return x**2,

# Set up the Genetic Algorithm framework
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 255)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    population = toolbox.population(n=300)
    ngen = 40
    cxpb = 0.5
    mutpb = 0.2

    # Apply the evolutionary algorithm
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=False)

    # Find the best individual in the final population
    top_individual = tools.selBest(population, 1)[0]
    return top_individual

best_individual = main()
print(f"Best individual: {best_individual[0]}")
print(f"Maximum value of f(x) = x^2: {best_individual[0]**2}")
