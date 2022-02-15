import numpy as np
import matplotlib.pyplot as plt
import pickle

N = 100
t_end = 1

D = 1
delta_t = 0.00001
delta_x = 1 / N

constant = D * delta_t / delta_x ** 2

grid = np.zeros((N, N))

for i in range(N):
    grid[0][i] = 1

grid_list = [grid]

# counter = 0
# for t in np.arange(0, t_end, delta_t):
#     counter += 1
#     print(f"{t:.2f}/{t_end}", end='\r')

#     new_grid = np.zeros((N, N))
#     for i in range(N):
#         new_grid[0][i] = 1

#     for y in range(1, N - 1):
#         for x in range(1, N - 1):
#             between_brackets = grid[y][x+1] + grid[y][x-1] + grid[y-1][x] + grid[y+1][x] - 4 * grid[y][x]

#             new_grid[y][x] = grid[y][x] + constant * between_brackets

#         new_grid[y][0] = grid[y][x] + constant * (grid[y][1] + grid[y][-2] + grid[y-1][0] + grid[y+1][0] - 4 * grid[y][0])
#         new_grid[y][-1] = grid[y][x] + constant * (grid[y][1] + grid[y][-2] + grid[y-1][-1] + grid[y+1][-1] - 4 * grid[y][-1])

#     grid = new_grid

#     if counter % 100 == 0:
#         grid_list.append(grid)

output = "1.2.pkl"
# with open(output, 'wb') as fp:
#     pickle.dump(grid_list, fp)

with open(output, 'rb') as fp:
    grid_list = pickle.load(fp)
grid = grid_list[0]

for grid in grid_list:
    plt.imshow(grid)
    plt.draw()
    plt.pause(0.001)
    plt.clf()