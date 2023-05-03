"""
Ant Colony Optimisation to Solve Travelling Salesman Problem(TSP)
"""

import copy
import numpy as np
import file_read

def totalDistance(adjacent_matrix, cityList):
    distance = 0
    for i in range(0, len(cityList[0])-1):
        distance = distance + adjacent_matrix[cityList[0][i]-1, cityList[0][i+1]-1]

    return distance

def local_search_2_opt(adjacent_matrix, cityList, improvement=-1):
   
    if (improvement < 0):
        count = -2
    else:
        count = 0
    cityList = copy.deepcopy(cityList)
    distance = cityList[1]*2
    while (count < improvement):
        best_route = copy.deepcopy(cityList)
        seed = copy.deepcopy(cityList)
        
        for i in range(0, len(cityList[0]) - 2):
            for j in range(i+1, len(cityList[0]) - 1):
                best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
                best_route[0][-1] = best_route[0][0]
                best_route[1] = totalDistance(adjacent_matrix, best_route)
                if (cityList[1] > best_route[1]):
                    cityList = copy.deepcopy(best_route)
                best_route = copy.deepcopy(seed)
        
        count = count + 1
        
        if (distance > cityList[1] and improvement < 0):
            distance = cityList[1]
            count = -2
            improvement = -1
        elif (cityList[1] >= distance and improvement < 0):
            count = -1
            improvement = -2

    return cityList[0], cityList[1]

def visibility(adjacent_matrix):
    eita = np.zeros((adjacent_matrix.shape[0], adjacent_matrix.shape[0]))
    
    for i in range(0, adjacent_matrix.shape[0]):
        for j in range(0, adjacent_matrix.shape[1]):
            if (i == j or adjacent_matrix[i, j] == 0):
                eita[i, j] = 0.000001
            else:
                eita[i, j] = 1/adjacent_matrix[i, j]

    return eita

def city_probability(h, thau, city=0, alpha=1, beta=2, cityList=[]):
    probability = np.zeros((h.shape[0], 3))
    
    for i in range(0, probability.shape[0]):
        if (i+1 not in cityList):
            probability[i, 0] = (thau[i, city]**alpha)*(h[i, city]**beta)
    
    for i in range(0, probability.shape[0]):
        if (i+1 not in cityList and probability[:, 0].sum() != 0):
            probability[i, 1] = probability[i, 0]/probability[:, 0].sum()
        if (i == 0):
            probability[i, 2] = probability[i, 1]
        else:
            probability[i, 2] = probability[i, 1] + probability[i - 1, 2]
   
    if (len(cityList) > 0):
        for i in range(0, len(cityList)):
            probability[cityList[i]-1, 2] = 0.0
    
    return probability

def selectCity(probability_matrix, cityList=[]):
    random = np.random.rand()
    city = 0
    for i in range(0, probability_matrix.shape[0]):
        if (random <= probability_matrix[i, 2] and i+1 not in cityList):
            city = i + 1
            break
    return city

def tabuList(adjacent_matrix, tabu, cityList=[]):
    distance = 0
    
    for i in range(0, len(cityList)-1):
        j = i + 1
        distance = distance + adjacent_matrix[cityList[i]-1, cityList[j]-1]
    
    pheromone = 1
    
    for i in range(0, len(cityList)-1):
        j = i + 1
        m = cityList[i]-1
        n = cityList[j]-1
        tabu[m, n] = tabu[m, n] + pheromone

    return tabu

def ants_path(adjacent_matrix, h, tabu, alpha, beta, full_list, ants):
    distance = np.sum(adjacent_matrix.sum())
    best_city_list = []
    best_path_distance = []
    
    for ant in range(1, ants+1):
        cityList = []
        cityList.append(ant)
        
        for i in range(0, adjacent_matrix.shape[0] - 1):
            probability = city_probability(
                h, tabu, city=i, alpha=alpha, beta=beta, cityList=cityList)
            path_point = selectCity(probability, cityList=cityList)
            if (path_point == 0):
                path_point = [
                    value for value in full_list if value not in cityList][0]
            cityList.append(path_point)
        cityList.append(cityList[0])
        path_distance = 0
        
        for i in range(0, len(cityList)-1):
            j = i + 1
            path_distance = path_distance + \
                adjacent_matrix[cityList[i]-1, cityList[j]-1]
        if (distance > path_distance):
            best_city_list = copy.deepcopy(cityList)
            best_path_distance = path_distance
            distance = path_distance
   
    best_route = copy.deepcopy([best_city_list])
    best_route.append(best_path_distance)
    
    best_city_list, best_path_distance = local_search_2_opt(
            adjacent_matrix, cityList=best_route, improvement=-1)
    
    tabu = tabuList(adjacent_matrix, tabu, cityList=best_city_list)
    
    return best_city_list, best_path_distance, tabu

def ACO(adjacent_matrix, ants=30, iterations=50, alpha=1, beta=10, decay=0.05, verbose=True):
    best_route = []
    full_list = list(range(1, adjacent_matrix.shape[0] + 1))
    distance = np.sum(adjacent_matrix.sum())
    eita = visibility(adjacent_matrix)
    tabu = np.ones((adjacent_matrix.shape[0], adjacent_matrix.shape[0]))
    
    for i in range(1, iterations+1):
        if (verbose == True and i % 5 == 0 ):
            print('Iteration = ', i, 'Distance = ', best_route[1])
        
        cityList, path_distance, tabu = ants_path(
            adjacent_matrix, eita, tabu, alpha, beta, full_list, ants)
        tabu = tabu*(1 - decay)
        
        if (distance > path_distance):
            best_route = copy.deepcopy([cityList])
            best_route.append(path_distance)
            distance = best_route[1]
        # print(distance)

    route, distance = best_route
    return route, distance

acoParameter = {
    'adjacent_matrix' : file_read.adj_matrix,
    'ants': 20,
    'iterations': 100,
    'alpha': 1,
    'beta': 2,
    'decay': 0.05,
    'verbose': True
}

route, distance = ACO(**acoParameter)
print("Optimal Route: ", route , "\n", "Distance: ", distance)

