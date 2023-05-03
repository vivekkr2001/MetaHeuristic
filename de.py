"""
Differential Evolution for Numerical Optimisation
Type- DE/rand/1/bin
"""

import numpy  as np
import random
import benchmarkFunctions

def function(x):
    d = len(x)
    sum = 0
    prod = 1
    for i in range(1, d+1):
        sum += x[i-1]**2 / 4000
        prod *= np.cos(x[i-1] / np.sqrt(i))
    return 1 + sum - prod

def initialPopulation(populationSize, dimension, minValue, maxValue, function):
    position = np.zeros((populationSize, dimension))
    for i in range(0, populationSize):
        for j in range(0, dimension):
            position[i, j] = random.uniform(minValue, maxValue)
        position[i, -1] = function(position[i, 0:position.shape[1]-1])
    return position

def mutation(position, best_global, j, k, l, F ):
    offspring = np.copy(best_global)
    for i in range(0, len(best_global)):
        offspring[i] = position[j,i] + F*(position[k, i] - position[l, i])
    
    return offspring

def discreteRecombination(position,mutated, dimension, best_global, k0, minValue, maxValue, Cr, function):
    child = np.copy(best_global)
    for i in range(0, len(best_global)):
        ri = np.random.rand()
        if (ri <= Cr):
            child[i] = mutated[i]   
        else:
            child[i] = position[k0, i]

        if (i < dimension and child[i] > maxValue):
            child[i] = maxValue
        elif(i < dimension and child[i] < minValue):
            child[i] = minValue

    child[-1] = function(child[0:dimension])
    return child


def DE(populationSize, dimension, minValue, maxValue, iterations, F , Cr, function, verbose = True):    
    position    = initialPopulation(populationSize, dimension, minValue, maxValue, function)
    best_global = np.copy(position [position [:,-1].argsort()][0,:])
    
    for i in range(1, iterations+1):
        if (verbose == True):
            print('Iteration = ', i, ' f(x) ', best_global[-1])
        
        for j in range(0, position.shape[0]):
            k = int(np.random.randint(position.shape[0], size = 1))
            l = int(np.random.randint(position.shape[0], size = 1))            
            while j == k  and k == l:
                k = int(np.random.randint(position.shape[0], size = 1))
                l = int(np.random.randint(position.shape[0], size = 1))
           
            mutated = mutation(position, best_global, j, k, l, F)
           
            vi = discreteRecombination(position,mutated,dimension, best_global, j, minValue, maxValue, Cr, function)        
            
            if (vi[-1] <= position[j,-1]):
                for m in range(0, position.shape[1]):
                    position[j,m] = vi[m]
            if (best_global[-1] > position [position [:,-1].argsort()][0,:][-1]):
                best_global = np.copy(position [position [:,-1].argsort()][0,:])  
    return best_global


griewank = {
    'populationSize': 125,
    'dimension': 10,
    'minValue': -600,
    'maxValue': 600,
    'iterations': 1500,
    'F': 0.6,
    'Cr': 0.2,
    'verbose': True,
    'function': benchmarkFunctions.griewank
}



rastrigin = {
    'populationSize': 125,
    'dimension': 10,
    'minValue': -15,
    'maxValue': 15,
    'iterations': 500,
    'verbose': False,
    'function': benchmarkFunctions.rastrigin
}

rosenbrock = {
    'populationSize': 125,
    'dimension': 10,
    'minValue': -15,
    'maxValue': 15,
    'iterations': 500,
    'verbose': False,
    'function': benchmarkFunctions.rosenbrock
}

ackley = {
    'populationSize': 125,
    'dimension': 10,
    'minValue': -32.768,
    'maxValue': 32.768,
    'iterations': 500,
    'verbose': True,
    'F': 0.6,
    'Cr': 0.2,
    'function': benchmarkFunctions.ackley
}

schwefel = {
    'populationSize': 125,
    'dimension': 10,
    'minValue': -500,
    'maxValue': 500,
    'iterations': 1500,
    'verbose': True,
    'F': 0.6,
    'Cr': 0.2,
    'function': benchmarkFunctions.schwefel
}

solution = DE(**griewank)
print("Solution: ", solution, "\n", "Value: ", solution[-1])