import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

N = 100
delta_x = 1 / N

D = 1
epsilon = 1e-3

eta = 2

def SOR(omega, grid, cluster):
    delta = np.inf
    delta_list = []
    counter = 0

    while delta > epsilon:
        counter += 1
        #print(counter)
        if counter == 5000:
            return counter, counter
        diff_list = []

        new_grid = np.zeros((N, N))
        new_grid[0] = 1

        for point in cluster:
            new_grid[point[1]][point[0]] = 0

        for y in range(1, N - 1):
            if [0, y] not in cluster:
                new_grid[y][0] = omega / 4 * (grid[y][1] + grid[y][-2] + grid[y + 1][0] + new_grid[y - 1][0]) + (1 - omega) * grid[y][0]
                diff_list.append(abs(new_grid[y][0] - grid[y][0]))

            for x in range(1, N - 1):
                if [x, y] not in cluster:
                    new_grid[y][x] = omega / 4 * (grid[y][x + 1] + new_grid[y][x - 1] + grid[y + 1][x] + new_grid[y - 1][x]) + (1 - omega) * grid[y][x]
                    diff_list.append(abs(new_grid[y][x] - grid[y][x]))

            if [N-1, y] not in cluster:
                new_grid[y][-1] = omega / 4 * (grid[y][1] + new_grid[y][-2] + grid[y + 1][-1] + new_grid[y - 1][-1]) + (1 - omega) * grid[y][-1]
                diff_list.append(abs(new_grid[y][-1] - grid[y][-1]))

        delta = max(diff_list)
        delta_list.append(delta)

        grid = new_grid

    return grid

grid = np.zeros((N,N))

grid[0] = 1

cluster = [[50, N-1]]


for t in range(250):
    print(t)

    grid = SOR(1.8, grid, cluster)

    candidates = []

    for point in cluster:
        x = point[0]
        y = point[1]
        if [x + 1, y] not in cluster and [x + 1, y] not in candidates and x + 1 < N:
            candidates.append([x + 1, y])
        if [x - 1, y] not in cluster and [x - 1, y] not in candidates and x - 1 >= 0:
            candidates.append([x - 1, y])
        if [x, y + 1] not in cluster and [x, y + 1] not in candidates and y + 1 < N:
            candidates.append([x, y + 1])
        if [x, y - 1] not in cluster and [x, y - 1] not in candidates and y - 1 >= 0:
            candidates.append([x, y - 1])

    concentration_list = []

    for point in candidates:
        concentration_list.append(grid[point[1]][point[0]]**eta)

    probability_list = []
    concentration_sum = sum(concentration_list)

    for i in range(len(concentration_list)):
        probability = concentration_list[i] / concentration_sum
        probability_list.append(probability)
    #print(cluster, candidates)


    new_point = np.random.choice(range(len(candidates)), p=probability_list)
    
    cluster.append(candidates[new_point])

for y in range(N):
    for x in range(N):
        if grid[y][x] == 0:
            grid[y][x] = None

plt.imshow(grid, cmap='gist_rainbow', norm=colors.LogNorm())
plt.show()



cluster_grid = np.zeros((N,N))
for point in cluster:
    cluster_grid[point[1]][point[0]] = 1

for point in candidates:
    cluster_grid[point[1]][point[0]] = 2
plt.imshow(cluster_grid)
plt.show()