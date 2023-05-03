import math
import numpy as np

def euclidean_distance(city1, city2):
    x1 = city1['latitude']
    x2 = city2['latitude']
    y1 = city1['longitude']
    y2 = city2['longitude']
    rij = math.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2)/10)
    tij = round(rij, 0)
    if(tij < rij):
        dij = tij + 1
    else:
        dij = tij
    
    return dij

filename = "./tspInstances/pr124.tsp"
with open(filename, 'r') as file:
    lines = file.readlines()
    cities = []  
    for line in lines[6::1]:  
        values = line.strip().split()
    
        if values[0] == "EOF":
            break;
        city = {'name': int(values[0]), 'longitude': float(values[1]), 'latitude': float(values[2])}
        cities.append(city)

num_cities = len(cities)
adj_matrix = np.asarray([[0 for _ in range(num_cities)] for _ in range(num_cities)])
city_number = [city['name'] for city in cities ]

# print(city_number)
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            dist = euclidean_distance(cities[i], cities[j])
            adj_matrix[i,j] = dist
            adj_matrix[j,i] = dist