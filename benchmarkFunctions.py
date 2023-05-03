import numpy as np
import math

def griewank(x):
    d = len(x)
    sum = 0
    prod = 1
    for i in range(1,d+1):
        sum += x[i-1]**2 / 4000
        prod *= np.cos(x[i-1] / np.sqrt(i))
    return 1 + sum - prod

def rastrigin(x):
    d = len(x)
    square = 0;
    multiplicate = 0;
    const = 0;
    for i in range(0,d):
        square += x[i]**2
        multiplicate += 10* np.cos(2*math.pi*x[i])
        const += 10
    
    return square - multiplicate + const

def rosenbrock(x):
    d = len(x)
    mult = 0
    sum = 0
    for i in range(0,d-1):
        sum += 100 * (x[i+1] - x[i]**2)**2
        mult += (1 - x[i])**2

    return sum + mult

def ackley(x):
    d = len(x)
    sum_sq_term = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(x)) / d))
    cos_term = -np.exp(np.sum(np.cos(2*math.pi * x)) / d)
    return 20 + np.exp(1) + sum_sq_term + cos_term

def schwefel(x):
    d = len(x)
    sum_term = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
    return 418.9829 * d + sum_term
