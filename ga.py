"""
Genetic Algorithm to Solve Travelling Salesman Problem(TSP)
"""

import copy
import numpy as np
import random
import file_read

def totalDistance(adjacent_matrix, cityList):
    distance = 0
    for i in range(0, len(cityList[0])-1):
        distance = distance + adjacent_matrix[cityList[0][i]-1, cityList[0][i+1]-1]

    return distance


def initial_population(population_size, adjacentMatrix):
    population = []

    for i in range(0, population_size):
        pop = [[], float("inf")]
        cityList = random.sample(
            list(range(1, adjacentMatrix.shape[0]+1)), adjacentMatrix.shape[0])
        pop[0] = cityList
        pop[1] = totalDistance(adjacentMatrix, pop)
        population.append(pop)

    return population

def fitness_function(population):
    fitness = np.zeros(len(population))
    for i in range(len(population)):
        fitness[i] = population[i][1]
    fit_sum = fitness.sum()
    
    for i in range(0, len(fitness)):
        fitness[i] = fitness[i]/fit_sum
    
    return fitness


def roulette_wheel(population, fitness):
    parents = []
    idx = np.random.randint(0, len(fitness))
    while len(parents) < 2:
        if fitness[idx] > np.random.rand():
            parents.append(population[idx][0])

    return parents


def ordered_crossover(parent_1, parent_2):
    point_1 = random.randint(0, len(parent_1)-1)
    point_2 = random.randint(0, len(parent_1)-1)

    if point_1 > point_2:
        point_1, point_2 = point_2, point_1

    child = [-1] * len(parent_1)
    child[point_1:point_2+1] = parent_1[point_1:point_2+1]

    for i in range(0, len(child)):
        if parent_2[i] not in child:
            for j in range(len(parent_1)):
                if child[j] == -1:
                    child[j] = parent_2[i]
                    break
    return child


def crossover(adjacentMatrix, population, fitness):
    new_population = []
    for i in range(len(population)):
        parents = roulette_wheel(population, fitness)
        newPopulation = [[], float('inf')]
        parent1 = parents[0]
        parent2 = parents[1]
        newPopulation[0] = ordered_crossover(parent1, parent2)
        newPopulation[1] = totalDistance(adjacentMatrix, newPopulation)

        if (newPopulation[1] > population[i][1]):
            newPopulation[0] = population[i][0]
            newPopulation[1] = population[i][1]

        new_population.append(newPopulation)

    return new_population

def two_opt_swap(distance_matrix, city_tour):
    i, j = random.sample(range(0, len(city_tour[0])-1), 2)
    new_tour = copy.deepcopy(city_tour)
    if (i > j):
        i, j = j, i
    new_tour[0][i:j+1] = list(reversed(new_tour[0][i:j+1]))
    new_tour[1] = totalDistance(distance_matrix, new_tour)

    return new_tour

def local_search(distance_matrix, city_tour, max_attempts=10, neighbourhood_size=5):
    count = 0
    solution = copy.deepcopy(city_tour)
    
    while (count < max_attempts):
        for i in range(0, neighbourhood_size):
            candidate = two_opt_swap(distance_matrix, solution)
        if candidate[1] < solution[1]:
            solution = copy.deepcopy(candidate)
            count = 0
        else:
            count = count + 1
    return solution

def swapMutation(adjacentMatrix, individual):
    k = random.sample(list(range(1, len(individual[0])-1)), 2)
    k1 = k[0]
    k2 = k[1]
    A = individual[0][k1]
    B = individual[0][k2]
    individual[0][k1] = B
    individual[0][k2] = A
    individual[1] = totalDistance(adjacentMatrix, individual)
   
    individual = local_search(adjacentMatrix, individual)
    
    return individual


def mutation(adjacentMatrix, newPopulation, mutation_rate):
    for i in range(len(newPopulation)):
        probability = np.random.rand()
        if (probability <= mutation_rate):
            newPopulation[i] = swapMutation(
                adjacentMatrix, newPopulation[i])
    return newPopulation


def genetic_algorithm(adjacentMatrix, population_size=40, mutation_rate=0.5, iterations=1000, verbose=True):
    population = initial_population(population_size, adjacentMatrix)
    cost = [item[1] for item in population]
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    bestSolution = population[0]
    fitness = fitness_function(population)
    count = 0

    for i in range(1, iterations+1):
        if (verbose == True and i % 5 == 0):
            print('Generation = ', i, 
                  'Distance = ', round(bestSolution[1], 2))

        newPopulation = crossover(adjacentMatrix, population, fitness)
        

        newPopulation = mutation(adjacentMatrix, newPopulation,
                             mutation_rate)
        
        cost = [item[1] for item in newPopulation]
        cost, population = (list(t) for t in zip(*sorted(zip(cost, newPopulation))))
        newBestSolution = population[0]
        fitness = fitness_function(population)
        if (bestSolution[1] > newBestSolution[1]):
            bestSolution = newBestSolution
        count = count + 1

    route, distance = bestSolution

    return route, distance


if __name__ == "__main__":
    genetic_algorithm(file_read.adj_matrix)
