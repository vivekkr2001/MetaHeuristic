"""
Particle Swarm Optimisation for Numerical Optimisation
Type:- Paricle Swarm Optimizer with inertia
x id (t) = f (w(t), x id (t − 1), v id (t − 1), p id , p gd )
v id (t) = w(t) ∗ v id (t − 1) + c 1 ϕ 1 (p id − x id (t − 1)) + c 2 ϕ 2 (p gd − x id (t − 1))
w(t) = ((T max − t) ∗ (w start − w end ))/Tmax + w_end
"""

import numpy as np
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


def initial_position(populationSize, dimension, minValue, maxValue, function):
    position = np.zeros((populationSize, dimension))
    for i in range(0, populationSize):
        for j in range(0, dimension):
            position[i, j] = random.uniform(minValue, maxValue)
        position[i, -1] = function(position[i, 0:position.shape[1]-1])
    return position


def initial_velocity(position, dimension, minValue, maxValue):
    velocity = np.zeros((position.shape[0], dimension))
    for i in range(0, velocity.shape[0]):
        for j in range(0, velocity.shape[1]):
            velocity[i, j] = random.uniform(minValue, maxValue)
    return velocity


def update_individual(position, individual):
    for i in range(0, position.shape[0]):
        if (individual[i, -1] > position[i, -1]):
            for j in range(0, position.shape[1]):
                individual[i, j] = position[i, j]
    return individual


def velocity_update(position, velocity, individual, best_global, w):
    phi1 = 2 
    phi2 = 2
    c1 = np.random.rand()
    c2 = np.random.rand()
    velocity = np.zeros((position.shape[0], velocity.shape[1]))

    for i in range(0, velocity.shape[0]):
        for j in range(0, velocity.shape[1]-1):
            velocity[i, j] = w*velocity[i, j] + phi1*c1 * \
                (individual[i, j] - position[i, j]) + \
                phi2*c2*(best_global[j] - position[i, j])
    return velocity


def update_position(position, velocity, minValue, maxValue, function):
    for i in range(0, position.shape[0]):
        for j in range(0, position.shape[1] - 1):
            position[i, j] = np.clip(
                (position[i, j] + velocity[i, j]),  minValue,  maxValue)
            
        position[i, -1] = function(position[i, 0:position.shape[1]-1])
    
    return position


def PSO(populationSize, dimension, minValue, maxValue, iterations, function, verbose):
    
    position = initial_position(populationSize, dimension, minValue, maxValue, function)
    velocity = initial_velocity(position, dimension, minValue, maxValue)
    
    individual = np.copy(position)
    best_global = position[position[:, -1].argmin()].copy()
    
    w_start = 0.9
    w_end = 0.4
    
    for i in range(1,iterations+1):
        if (verbose == True and i % 5 == 0):
            print('Iteration= ', i, ' f(x) = ', best_global[-1])

        position = update_position(position, velocity, minValue, maxValue, function)
        individual = update_individual(position, individual)
        value = np.copy(individual[individual[:, -1].argsort()][0, :])
        
        if (best_global[-1] > value[-1]):
            best_global = np.copy(value)

        w = w_end + ((iterations - (i-1)) * (w_start - w_end))/iterations
        velocity = velocity_update(position, velocity, individual, best_global, w)

    return best_global


griewank = {
    'populationSize': 125,
    'dimension': 10,
    'minValue': -600,
    'maxValue': 600,
    'iterations': 500,
    'verbose': True,
    'function': benchmarkFunctions.griewank
}


rastrigin = {
    'populationSize': 125,
    'dimension': 10,
    'minValue': -15,
    'maxValue': 15,
    'iterations': 1000,
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
    'verbose': False,
    'function': benchmarkFunctions.ackley
}

schwefel_10 = {
    'populationSize': 125,
    'dimension': 10,
    'minValue': -500,
    'maxValue': 500,
    'iterations': 500,
    'verbose': True,
    'function': benchmarkFunctions.schwefel
}

solution = PSO(**griewank)
print("Solution: ", solution, "\n", "Value: ", solution[-1])
