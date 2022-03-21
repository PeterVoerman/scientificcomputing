import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs

rows = 100
cols = 100
n_elements = rows * cols

grid = np.zeros((rows, cols), dtype=int)


counter = 0
for y in range(rows):
    for x in range(cols):
        grid[y][x] = int(counter)
        counter += 1

matrix = np.zeros((n_elements, n_elements))

for y in range(0, rows):
    for x in range(0, cols):
        element = grid[y][x]
        if y > 0:
            matrix[grid[y-1][x]][element] = 1
            matrix[element][element] -= 1
        if y < rows - 1:
            matrix[grid[y+1][x]][element] = 1
            matrix[element][element] -= 1
        if x > 0:
            matrix[grid[y][x-1]][element] = 1
            matrix[element][element] -= 1
        if x < cols - 1:
            matrix[grid[y][x+1]][element] = 1
            matrix[element][element] -= 1
        
print(matrix)

# solution = linalg.eig(matrix)

# print(solution[0])
# print(solution[1][0])

# solution = eigs(matrix)

# print(solution[0])
# print(solution[1][0])

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

mask = create_circular_mask(11, 11)
print(mask)