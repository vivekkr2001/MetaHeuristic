"""
Artificial Bee Colony for Numerical Optimisation
Type:- Artificial Colony System(ACS)
"""

import numpy as np
import random
import benchmarkFunctions

def generateSources(foodSources,dimension, minValue, maxValue, function):
    sources = np.zeros((foodSources, dimension))
    for i in range(0, foodSources):
        for j in range(0, dimension):
            sources[i, j] = random.uniform(minValue, maxValue)
        sources[i, -1] = function(sources[i, 0:sources.shape[1]-1])
    return sources

def fitness_calc(function_value):
    if (function_value >= 0):
        fitness_value = 1.0/(1.0 + function_value)
    else:
        fitness_value = 1.0 + abs(function_value)
    return fitness_value

def fitness_function(sourceSearch):
    fitness = np.zeros((sourceSearch.shape[0], 2))

    for i in range(0, fitness.shape[0]):
        fitness[i, 0] = fitness_calc(sourceSearch[i, -1])
    
    fit_sum = fitness[:, 0].sum()
    fitness[0, 1] = fitness[0, 0]
    
    for i in range(1, fitness.shape[0]):
        fitness[i, 1] = (fitness[i, 0] + fitness[i-1, 1])
    
    for i in range(0, fitness.shape[0]):
        fitness[i, 1] = fitness[i, 1]/fit_sum
    
    return fitness

def selectByFitness(fitness):
    ix = 0
    random = np.random.rand()
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
            ix = i
            break
    return ix

def employedBee(sources,dimension, minValue, maxValue, function):
    sourceSearch = np.copy(sources)
    new_solution = np.zeros((1, dimension))
    trial = np.zeros((sources.shape[0], 1))
    for i in range(0, sourceSearch.shape[0]):
        phi = random.uniform(-1, 1)
        j = np.random.randint(dimension, size=1)[0]
        k = np.random.choice([idx for idx in range(sourceSearch.shape[0]) if idx != i])

        xij = sourceSearch[i, j]
        xkj = sourceSearch[k, j]
        vij = xij + phi*(xij - xkj)

        for variable in range(0, dimension):
            new_solution[0, variable] = sourceSearch[i, variable]

        new_solution[0, j] = np.clip(vij, minValue, maxValue)
        new_function_value = function(
            new_solution[0, 0:new_solution.shape[1]])
        if (fitness_calc(new_function_value) > fitness_calc(sourceSearch[i, -1])):
            sourceSearch[i, j] = new_solution[0, j]
            sourceSearch[i, -1] = new_function_value
        else:
            trial[i, 0] = trial[i, 0] + 1
        for variable in range(0, dimension):
            new_solution[0, variable] = 0.0
    return sourceSearch, trial

def onlookerBee(sourceSearch, dimension, fitness, trial, minValue, maxValue, function):
    sourceImprovement = np.copy(sourceSearch)
    new_solution = np.zeros((1, dimension))
    trial_update = np.copy(trial)
    for _ in range(0, sourceImprovement.shape[0]):
        i = selectByFitness(fitness)
        phi = random.uniform(-1, 1)
        j = np.random.randint(dimension, size=1)[0]
        k = np.random.choice([idx for idx in range(sourceImprovement.shape[0]) if idx != i])

        xij = sourceImprovement[i, j]
        xkj = sourceImprovement[k, j]
        vij = xij + phi*(xij - xkj)
        for variable in range(0, dimension):
            new_solution[0, variable] = sourceImprovement[i, variable]
        new_solution[0, j] = np.clip(vij,  minValue, maxValue)
        new_function_value = function(
            new_solution[0, 0:new_solution.shape[1]])
        if (fitness_calc(new_function_value) > fitness_calc(sourceImprovement[i, -1])):
            sourceImprovement[i, j] = new_solution[0, j]
            sourceImprovement[i, -1] = new_function_value
            trial_update[i, 0] = 0
        else:
            trial_update[i, 0] = trial_update[i, 0] + 1
        for variable in range(0, dimension):
            new_solution[0, variable] = 0.0
    return sourceImprovement, trial_update

def scoutBee(sourceImprovement, trial_update, limit, function):
    for i in range(0, sourceImprovement.shape[0]):
        if (trial_update[i, 0] > limit):
            for j in range(0, sourceImprovement.shape[1] - 1):
                sourceImprovement[i, j] = np.random.normal(0, 1, 1)[0]
            function_value = function(
                sourceImprovement[i, 0:sourceImprovement.shape[1]-1])
            sourceImprovement[i, -1] = function_value

    return sourceImprovement

def ABC(foodSources,dimension, iterations, minValue, maxValue, employed_bees, outlookers_bees, limit, function, verbose):
    best_value = float('inf')
    sources = generateSources(foodSources,dimension, minValue, maxValue, function)
    fitness = fitness_function(sources)

    for i in range(1, iterations+1):
        if (verbose == True and i % 5 == 0 and i > 0):
            print('Iteration = ', i, '\n f(x) = ',best_solution, "\n Best Value: ", best_value)
        empBee = employedBee(sources,dimension, minValue, maxValue, function)
        
        for i in range(0, employed_bees - 1):
            empBee = employedBee(empBee[0], dimension, minValue, maxValue, function)
        fitness = fitness_function(empBee[0])

        onLookBee = onlookerBee(empBee[0], dimension, fitness, empBee[1], minValue, maxValue, function)
        
        for i in range(0, outlookers_bees - 1):
            onLookBee = onlookerBee(onLookBee[0], dimension, fitness, onLookBee[1], minValue, maxValue, function)
        value = np.copy(onLookBee[0][onLookBee[0][:, -1].argsort()][0, :])
        
        if (best_value > value[-1]):
            best_solution = np.copy(value)
            best_value = np.copy(value[-1])
        
        sources = scoutBee(onLookBee[0], onLookBee[1], limit=limit, function=function)
        
        fitness = fitness_function(sources)
    return best_solution, best_value

griewank = {
    'foodSources': 125,
      'dimension': 10, 
    'minValue': -600,
     'maxValue': 600,
     'iterations': 10,
     'employed_bees': 20, 
     'outlookers_bees': 20,
     'limit': 3, 
     'function': benchmarkFunctions.griewank,
     'verbose' : True
}
solution, value = ABC(**griewank)
print("Solution: ", solution, "\n", "Best Value= ", value)

rastrigin = {
    'foodSources': 125,
      'dimension': 10, 
    'minValue': -15,
     'maxValue': 15,
     'iterations': 10,
     'employed_bees': 20, 
     'outlookers_bees': 20,
     'limit': 3, 
     'function': benchmarkFunctions.rastrigin,
     'verbose' : False
}
# solution, value = ABC(**rastrigin)
# print("Solution: ", solution, "\n", "Best Value= ", value)

rosenbrock = {
    'foodSources': 125,
      'dimension': 10, 
    'minValue': -15,
     'maxValue': 15,
     'iterations': 10,
     'employed_bees': 20, 
     'outlookers_bees': 20,
     'limit': 3, 
     'function': benchmarkFunctions.rosenbrock,
     'verbose' : False
}
solution, value = ABC(**rosenbrock)
print("Solution: ", solution, "\n", "Best Value= ", value)

ackley = {
    'foodSources': 125,
      'dimension': 10, 
    'minValue': -32.768,
     'maxValue': 32.768,
     'iterations': 100,
     'employed_bees': 20, 
     'outlookers_bees': 20,
     'limit': 3, 
     'function': benchmarkFunctions.ackley,
     'verbose' : False
}
solution, value = ABC(**ackley)
print("Solution: ", solution, "\n", "Best Value= ", value)

schwefel = {
    'foodSources': 125,
      'dimension': 10, 
    'minValue': -500,
     'maxValue': 500,
     'iterations': 100,
     'employed_bees': 20, 
     'outlookers_bees': 20,
     'limit': 3, 
     'function': benchmarkFunctions.schwefel,
     'verbose' : False
}
solution, value = ABC(**rastrigin)
print("Solution: ", solution, "\n", "Best Value= ", value)