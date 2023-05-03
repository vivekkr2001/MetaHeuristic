"""
1-max Code for MetaHeuristics
"""

import random
import copy


def calculate_fitness(individual):
    return sum(individual) 

def sort_by_fitness(population):
    fp = []
    for individual in population:
        fitness = calculate_fitness(individual)
        fp.append((fitness, individual))
    fp.sort(reverse=True)  

    sorted_population = []
    for fitness,  individual in fp:
        sorted_population.append(individual)
    return sorted_population

def selection(population, population_size, selection_rate):
    sorted_population = sort_by_fitness(population)
    n = int(population_size * selection_rate)
    return sorted_population[0 : n]

def crossover(ind1, ind2, chromosomes_size):
    r1 = random.randint(0, chromosomes_size -1)
    r2 = random.randint(r1 + 1, chromosomes_size)
    ind = copy.deepcopy(ind1)
    ind[r1:r2] = ind2[r1:r2]
    return ind

def mutation(ind1, chromosome_size, mutation_rate):
    ind2 = copy.deepcopy(ind1)
    for i in range(chromosome_size):
        if random.random() < mutation_rate:
            ind2[i] =  random.randint(0,1)
    return ind2

def main(population_size, chromosomes_size, iterations, mutation_rate=0.2, selection_rate = 0.5):
    population = []
    for i in range(population_size):
        individual =  []
        for j in range(chromosomes_size):
            individual.append(random.randint(0,1))
        population.append(individual)

    for generation in range(iterations):
        population = selection(population, population_size, selection_rate)  
        n = population_size - len(population)

        for i in range(n):
            parent1_index = random.randint(0, len(population) -1)
            parent2_index = random.randint(0, len(population) -1)

            individual = crossover(population[parent1_index], population[parent2_index], chromosomes_size)
            individual = mutation(individual, chromosomes_size, mutation_rate)
            population.append(individual)
        
        # if calculate_fitness(population[0]) is chromosomes_size:
        #     print("Best Found at "+ str(generation) + " Fitness: " + str(chromosomes_size))
        #     break;
    
    for individual in population:
        print("Individual: ", individual,"Fitness: ", calculate_fitness(individual))

if __name__ == "__main__":
    main(population_size=10, chromosomes_size=10, iterations=1000, mutation_rate=0.2, selection_rate=0.5)