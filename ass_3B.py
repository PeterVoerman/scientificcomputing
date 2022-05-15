# Nik Brouw and Peter Voerman
# Scientific computing exercise set 3

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors


def diffusion_solver(grid, x_length, y_length, filename, sparce=True, circle_list=[]):
    n_elements = x_length * y_length
    matrix = np.zeros((n_elements, n_elements))

    for y in range(0, y_length):
        for x in range(0, x_length):
            element = grid[y][x]
            if y > 0:
                matrix[grid[y-1][x]][element] = -1
            if y < y_length - 1 :
                matrix[grid[y+1][x]][element] = -1
            if x > 0:
                matrix[grid[y][x-1]][element] = -1
            if x < x_length - 1:
                matrix[grid[y][x+1]][element] = -1

    for y in range(y_length):
        for x in range(x_length):
            element = grid[y][x]
            if circle_list != [] and element not in circle_list:
                matrix[element] = 0

    np.fill_diagonal(matrix, 4)

    b = np.zeros(n_elements)

    b[int(0.3*y_length + 0.4 * y_length**2)] = 1

    solution = linalg.solve(matrix, b)

    solution_grid = np.zeros((y_length, x_length))

    for y in range(y_length):
        for x in range(x_length):
            solution_grid[y][x] = (solution[y * x_length + x])
    plt.xticks([0, x_length/4, x_length/2, x_length*.75,x_length-1], [0, 0.5, 1, 1.5, 2])
    plt.yticks([0, x_length/4, x_length/2, x_length*.75,x_length-1], [2, 1.5, 1, 0.5,0])
    plt.title("Diffusion on a circular disk")
    plt.imshow(solution_grid)
    plt.savefig(f"plots_3/diffusion/{filename}")

    return solution


def create_circular_mask(diameter):

    center = int(diameter/2)
    radius = center

    mask = [[False for i in range(diameter)] for i in range(diameter)]
    
    for y in range(diameter):
        for x in range(diameter):
            if np.sqrt((x-center) ** 2 + (y-center) ** 2) <= radius:
                mask[y][x] = True

    return mask

mask = create_circular_mask(100)

x_length = 100
y_length = 100

grid = np.zeros((y_length, x_length), dtype=int)

counter = 0
circle_list = []

for y in range(y_length):
    for x in range(x_length):
        if mask[y][x]:
            circle_list.append(counter)
        grid[y][x] = counter
        counter += 1

solution = diffusion_solver(grid, x_length, y_length, "circle", circle_list=circle_list)