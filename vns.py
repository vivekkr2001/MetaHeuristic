"""
Variable Neighbourhood Search with VND to Solve Travelling Salesman Problem(TSP)
"""

import random
import copy
import file_read

def totalDistance(adjacent_matrix, cityList):
    distance = 0
    for i in range(0, len(cityList[0])-1):
        distance = distance + adjacent_matrix[cityList[0][i]-1, cityList[0][i+1]-1]

    return distance

def initialPopulation(adjacent_matrix):
    cityList = [[], float('inf')]
    cities = random.sample(
        list(range(1, adjacent_matrix.shape[0]+1)), adjacent_matrix.shape[0])
    # cities.append(cities[0])
    cityList[0] = cities
    cityList[1] = totalDistance(adjacent_matrix, cityList)
    return cityList

def three_opt(cityList, i, j):
    candidate_tour = copy.deepcopy(cityList)
    n = len(cityList[0])-1
    
    k = random.choice([k for k in range(n) if k != i and k != j])
    if k > j:
        a, b, c = candidate_tour[0][i:j+1], candidate_tour[0][j+1:k+1], candidate_tour[0][k+1:]
        if random.random() < 0.5:
            candidate_tour[0][i:] = a + b + c
        else:
            candidate_tour[0][i:] = a + c + b
    elif i < k <= j:
        a, b, c = candidate_tour[0][i:k+1], candidate_tour[0][k+1:j+1], candidate_tour[0][j+1:]
        if random.random() < 0.5:
            candidate_tour[0][i:] = a + b + c
        else:
            candidate_tour[0][i:] = a + c + b
    elif k <= i:
        a, b, c = candidate_tour[0][k:i+1], candidate_tour[0][i+1:j+1], candidate_tour[0][j+1:]
        if random.random() < 0.5:
            candidate_tour[0][k:] = c + b + a
        else:
            candidate_tour[0][k:] = b + a + c
    
    return candidate_tour

def four_opt_swap(cityList, i, j):
    k, l = random.sample(range(0, len(cityList[0])-1), 2)

    new_tour = copy.deepcopy(cityList)
    if (i > j):
        i, j = j, i
    if (j > k):
        j, k = k, j
    if (k > l):
        k, l = l, k
    if (i > j):
        i, j = j, i
    if (j > k):
        j, k = k, j
    if (i > j):
        i, j = j, i
    new_tour[0][i:j+1] = list(reversed(new_tour[0][i:j+1]))
    new_tour[0][j:k+1] = list(reversed(new_tour[0][j:k+1]))
    new_tour[0][k:l+1] = list(reversed(new_tour[0][k:l+1]))
    return new_tour

def shakingProcedure(adjacent_matrix, cityList):
    new_tour = copy.deepcopy(cityList)
    i, j = random.sample(range(0, len(cityList[0])-1), 2)
    if (i > j):
        i, j = j, i
    new_tour[0][i:j+1] = list(reversed(new_tour[0][i:j+1]))
    new_tour[1] = totalDistance(adjacent_matrix, new_tour)
    return new_tour

def explore_neighborhood(adjacent_matrix, cityList, neighborhood_size):
    candidate_tour = copy.deepcopy(cityList)
    i, j = random.sample(range(0, len(cityList[0])-1), 2)
    if i > j:
        i, j = j, i

    if neighborhood_size == 2:
        candidate_tour[0][i:j+1] = list(reversed(candidate_tour[0][i:j+1]))
    elif neighborhood_size == 3:
        candidate_tour = three_opt(candidate_tour, i, j)
    elif neighborhood_size == 4:
        candidate_tour = four_opt_swap(candidate_tour, i, j)

    # candidate_tourq[0][-1] = candidate_tour[0][0]
    candidate_tour[1] = totalDistance(adjacent_matrix, candidate_tour)
    return candidate_tour

def vnd(adjacent_matrix, cityList, max_attempts=50, neighborhoods=[2, 3, 4]):
    best_tour = copy.deepcopy(cityList)
    count = 0
    k = 0
    
    while count < max_attempts:
        candidate_tour = copy.deepcopy(best_tour)
        neighborhood_size = neighborhoods[k % len(neighborhoods)]
        candidate_tour = explore_neighborhood(adjacent_matrix, candidate_tour, neighborhood_size)
        if candidate_tour[1] < best_tour[1]:
            best_tour = copy.deepcopy(candidate_tour)
            count = 0
            k = 0
        else:
            count += 1
            k += 1
    return best_tour

def VNS(adjacent_matrix, max_attempts=20, neighbourhood_size=10, iterations=50, verbose=True):
    initialCities = initialPopulation(adjacent_matrix)
    solution = copy.deepcopy(initialCities)
    best_solution = copy.deepcopy(initialCities)
    
    for i in range(iterations):
        if (verbose == True and i % 10 == 0):
            print('Iteration = ', i, 'Solution = ', best_solution)

        for i in range(0, neighbourhood_size):
            for j in range(0, neighbourhood_size):
                solution = shakingProcedure(adjacent_matrix, best_solution)

            solution = vnd(adjacent_matrix, solution, max_attempts, [2,3,4])

            if (solution[1] < best_solution[1]):
                best_solution = copy.deepcopy(solution)
                break

    route, distance = best_solution
    return route, distance

vnsParameter = {
    'adjacent_matrix' : file_read.adj_matrix,
    'neighbourhood_size': 20,
    'iterations': 500,
    'verbose': False
}

route, distance = VNS(**vnsParameter)
print("Optimal Route: ", route , "\n", "Distance: ", distance)

