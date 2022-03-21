import wave
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs

import matplotlib.pyplot as plt

def wave_solver(grid, x_length, y_length, filename, sparce=True):
    n_elements = x_length * y_length
    matrix = np.zeros((n_elements, n_elements))

    for y in range(0, y_length):
        for x in range(0, x_length):
            element = grid[y][x]
            if y > 0 and grid[y-1][x] != 0:
                matrix[grid[y-1][x]][element] = 1
                matrix[element][element] -= 1
            if y < y_length - 1 and grid[y+1][x] != 0:
                matrix[grid[y+1][x]][element] = 1
                matrix[element][element] -= 1
            if x > 0 and grid[y][x-1] != 0:
                matrix[grid[y][x-1]][element] = 1
                matrix[element][element] -= 1
            if x < x_length - 1 and grid[y][x+1] != 0:
                matrix[grid[y][x+1]][element] = 1
                matrix[element][element] -= 1

    print(matrix)
    if sparce:
        solution = eigs(matrix)
    else:
        solution = linalg.eig(matrix)


    for L in range(len(solution[1][0])):

        solution_grid = np.zeros((y_length, x_length))

        for y in range(y_length):
            for x in range(x_length):
                solution_grid[y][x] = abs(solution[1][y * y_length + x][L])

        plt.title(f"L={solution[0][L]}")
        plt.imshow(solution_grid)
        plt.savefig(f"plots_3/{filename}{L+1}")


x_length = 100
y_length = 100

grid = np.zeros((y_length, x_length), dtype=int)

counter = 0
for y in range(y_length):
    for x in range(x_length):
        grid[y][x] = int(counter)
        counter += 1

#wave_solver(grid, x_length, y_length, "square")

x_length = 4
y_length = 3

grid = np.zeros((y_length, x_length), dtype=int)



counter = 0
for y in range(y_length):
    for x in range(x_length):
        grid[y][x] = int(counter)
        counter += 1

print(grid)

wave_solver(grid, x_length, y_length, "rectangle")

array = np.zeros((50, 50))

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

mask = create_circular_mask(100, 100)

x_length = 100
y_length = 100

grid = np.zeros((y_length, x_length), dtype=int)

counter = 0
for y in range(y_length):
    for x in range(x_length):
        if mask[y][x]:
            grid[y][x] = int(counter)
            counter += 1

#wave_solver(grid, x_length, y_length, "circle")