import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

N = 50
delta_x = 1 / N

D = 1
epsilon = 1e-5

def jacobian():

    grid = np.zeros((N, N))

    for i in range(N):
        grid[0][i] = 1

    delta = np.inf
    delta_list = []
    counter = 0

    while delta > epsilon:
        counter += 1
        diff_list = []

        new_grid = np.zeros((N, N))
        for i in range(N):
            new_grid[0][i] = 1

        for y in range(1, N - 1):
            for x in range(1, N - 1):
                new_grid[y][x] = 0.25 * (grid[y][x + 1] + grid[y][x - 1] + grid[y + 1][x] + grid[y - 1][x])
                diff_list.append(abs(new_grid[y][x] - grid[y][x]))

            new_grid[y][0] = 0.25 * (grid[y][1] + grid[y][-2] + grid[y + 1][0] + grid[y - 1][0])
            new_grid[y][-1] = 0.25 * (grid[y][1] + grid[y][-2] + grid[y + 1][-1] + grid[y - 1][-1])

            diff_list.append(abs(new_grid[y][0] - grid[y][0]))
            diff_list.append(abs(new_grid[y][-1] - grid[y][-1]))

        delta = max(diff_list)
        delta_list.append(delta)

        grid = new_grid

    return grid, delta_list

def gauss_seidel():
    grid = np.zeros((N, N))

    for i in range(N):
        grid[0][i] = 1

    delta = np.inf
    delta_list = []
    counter = 0

    while delta > epsilon:
        counter += 1
        diff_list = []

        new_grid = np.zeros((N, N))
        for i in range(N):
            new_grid[0][i] = 1

        for y in range(1, N - 1):
            new_grid[y][0] = 0.25 * (grid[y][1] + grid[y][-2] + grid[y + 1][0] + new_grid[y - 1][0])

            for x in range(1, N - 1):
                new_grid[y][x] = 0.25 * (grid[y][x + 1] + new_grid[y][x - 1] + grid[y + 1][x] + new_grid[y - 1][x])
                diff_list.append(abs(new_grid[y][x] - grid[y][x]))

            
            new_grid[y][-1] = 0.25 * (grid[y][1] + new_grid[y][-2] + grid[y + 1][-1] + new_grid[y - 1][-1])

            diff_list.append(abs(new_grid[y][0] - grid[y][0]))
            diff_list.append(abs(new_grid[y][-1] - grid[y][-1]))

        delta = max(diff_list)
        delta_list.append(delta)

        grid = new_grid

    return grid, delta_list

def SOR(omega):
    grid = np.zeros((N, N))

    for i in range(N):
        grid[0][i] = 1

    delta = np.inf
    delta_list = []
    counter = 0

    while delta > epsilon:
        counter += 1
        diff_list = []

        new_grid = np.zeros((N, N))
        for i in range(N):
            new_grid[0][i] = 1

        for y in range(1, N - 1):
            new_grid[y][0] = omega / 4 * (grid[y][1] + grid[y][-2] + grid[y + 1][0] + new_grid[y - 1][0]) + (1 - omega) * grid[y][0]

            for x in range(1, N - 1):
                new_grid[y][x] = omega / 4 * (grid[y][x + 1] + new_grid[y][x - 1] + grid[y + 1][x] + new_grid[y - 1][x]) + (1 - omega) * grid[y][x]
                diff_list.append(abs(new_grid[y][x] - grid[y][x]))

            
            new_grid[y][-1] = omega / 4 * (grid[y][1] + new_grid[y][-2] + grid[y + 1][-1] + new_grid[y - 1][-1]) + (1 - omega) * grid[y][-1]

            diff_list.append(abs(new_grid[y][0] - grid[y][0]))
            diff_list.append(abs(new_grid[y][-1] - grid[y][-1]))

        delta = max(diff_list)
        delta_list.append(delta)

        grid = new_grid

    return grid, delta_list

def SOR_with_objects(omega, object_list):
    object_coords = []

    for object in object_list:
        x1, y1, x2, y2 = object

        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                object_coords.append((x, y))

    grid = np.zeros((N, N))

    for i in range(N):
        grid[0][i] = 1

    delta = np.inf
    delta_list = []
    counter = 0

    while delta > epsilon:
        counter += 1
        diff_list = []

        new_grid = np.zeros((N, N))
        for i in range(N):
            new_grid[0][i] = 1

        for y in range(1, N - 1):
            if (x, y) in object_coords:
                new_grid[y][x] = 0
            else:
                new_grid[y][0] = omega / 4 * (grid[y][1] + grid[y][-2] + grid[y + 1][0] + new_grid[y - 1][0]) + (1 - omega) * grid[y][0]

            for x in range(1, N - 1):
                if (x, y) in object_coords:
                    new_grid[y][x] = 0
                else:
                    new_grid[y][x] = omega / 4 * (grid[y][x + 1] + new_grid[y][x - 1] + grid[y + 1][x] + new_grid[y - 1][x]) + (1 - omega) * grid[y][x]
                diff_list.append(abs(new_grid[y][x] - grid[y][x]))

            if (x, y) in object_coords:
                new_grid[y][x] = 0
            else:
                new_grid[y][-1] = omega / 4 * (grid[y][1] + new_grid[y][-2] + grid[y + 1][-1] + new_grid[y - 1][-1]) + (1 - omega) * grid[y][-1]

            diff_list.append(abs(new_grid[y][0] - grid[y][0]))
            diff_list.append(abs(new_grid[y][-1] - grid[y][-1]))

        delta = max(diff_list)
        delta_list.append(delta)

        grid = new_grid

    return grid, delta_list

def analytical_solution(t, D = 1):
    solution_list = []

    for y in np.arange(0, 1.05, 0.05):
        solution = 0

        for i in range(100000):
            solution += math.erfc((1 - y + 2 * i) / (2 * math.sqrt(D * t))) - math.erfc((1 + y + 2 * i) / (2 * math.sqrt(D * t)))

        solution_list.append(solution)

    return solution_list

# jacobian_grid, jacobian_delta_list = jacobian()
# gauss_grid, gauss_delta_list = gauss_seidel()
# sor_grid, sor_delta_list = SOR(1.8)
# sor_grid_2, sor_delta_list_2 = SOR(1.7)
# sor_grid_3, sor_delta_list_3 = SOR(1.9)
# sor_grid_4, sor_delta_list_4 = SOR(1.5)

# plt.imshow(jacobian_grid)
# plt.show()

# plt.imshow(gauss_grid)
# plt.show()

# plt.imshow(sor_grid)
# plt.show()

# analytical_slice = analytical_solution(1)

# jacobian_slice = []
# gauss_slice = []
# sor_slice = []
# y_list = []

# for y in range(len(jacobian_grid)):
#     jacobian_slice.insert(0, jacobian_grid[y][25])
#     gauss_slice.insert(0, gauss_grid[y][25])
#     sor_slice.insert(0, sor_grid[y][25])

#     y_list.append(y / len(jacobian_grid))

# plt.plot(y_list, jacobian_slice, label="jacobian")
# plt.plot(y_list, gauss_slice, label="gauss")
# plt.plot(y_list, sor_slice, label="sor")
# plt.plot(np.arange(0, 1.05, 0.05), analytical_slice, label="analytical")
# plt.legend()
# plt.show()

# plt.plot(range(len(jacobian_delta_list)), jacobian_delta_list)
# plt.yscale('log')
# plt.show()

# plt.plot(range(len(gauss_delta_list)), gauss_delta_list)
# plt.yscale('log')
# plt.show()

# plt.plot(range(len(sor_delta_list)), sor_delta_list)
# plt.yscale('log')
# plt.show()

# plt.plot(range(len(sor_delta_list_2)), sor_delta_list_2)
# plt.yscale('log')
# plt.show()

# plt.plot(range(len(sor_delta_list_3)), sor_delta_list_3)
# plt.yscale('log')
# plt.show()

# plt.plot(range(len(sor_delta_list_4)), sor_delta_list_4)
# plt.yscale('log')
# plt.show()

# delta_list = []
# N_list = []
# omega_list = []

# for omega in np.arange(1.7, 1.95, 0.05):
#     print(omega)
#     grid, delta = SOR(omega)
#     delta_list.append(delta)
#     N_list.append(len(delta))
#     omega_list.append(omega)

# plt.plot(omega_list, N_list)
# plt.show()

grid, delta_list = SOR_with_objects(1.8, [(10, 5, 20, 20)])
plt.imshow(grid)
plt.show()
