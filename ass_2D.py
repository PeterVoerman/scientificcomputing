# Scientific computing Exercise 1.2
# Peter Voerman and Nik brouw

# The results of one of the simulations are stored in "1.2.pkl"
# Change the variable below to True in order to run the entire simulation again

run_again = False

import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import random

N = 100
t_end = 500

D = 0.08
delta_t = 1
delta_x = 1

f = 0.035
k = 0.06

output = "2withnoise.pkl"



def calculate_diffusion():
    u_grid = [[0.5 for i in range(N)] for i in range(N)]
    v_grid = np.zeros((N, N))

    for x in range(45, 55):
        for y in range(45, 55):
            v_grid[y][x] = 0.25

    u_grid_list = [u_grid]
    v_grid_list = [v_grid]


    counter = 0
    for t in np.arange(0, t_end, delta_t):
        counter += 1
        print(f"{t:.2f}/{t_end}", end='\r')

        new_u_grid = np.zeros((N, N))
        new_v_grid = np.zeros((N, N))

        for y in range(1, N - 1):
            for x in range(1, N - 1):
                new_u_grid[y][x] = delta_t * (D * ((u_grid[y][x+1]+u_grid[y][x-1]+u_grid[y+1][x]+u_grid[y-1][x]-4*u_grid[y][x])/delta_x**2)-u_grid[y][x]*v_grid[y][x]**2+f*(1-u_grid[y][x]))+u_grid[y][x]
                new_v_grid[y][x] = delta_t * (D * ((v_grid[y][x+1]+v_grid[y][x-1]+v_grid[y+1][x]+v_grid[y-1][x]-4*v_grid[y][x])/delta_x**2)+u_grid[y][x]*v_grid[y][x]**2-(f+k)*v_grid[y][x])+v_grid[y][x]

        for x in range(1, N - 1):
            new_u_grid[0][x] = delta_t * (D*((u_grid[0][x+1]+u_grid[0][x-1]+u_grid[1][x]-4*u_grid[0][x])/delta_x**2)-u_grid[0][x]*v_grid[0][x]**2+f*(1-u_grid[0][x]))+u_grid[0][x]
            new_v_grid[0][x] = delta_t * (D*((v_grid[0][x+1]+v_grid[0][x-1]+v_grid[1][x]-4*v_grid[0][x])/delta_x**2)+u_grid[0][x]*v_grid[0][x]**2-(f+k)*v_grid[0][x])+v_grid[0][x]

            new_u_grid[-1][x] = delta_t * (D*((u_grid[-1][x+1]+u_grid[-1][x-1]+u_grid[-2][x]-4*u_grid[-1][x])/delta_x**2)-u_grid[-1][x]*v_grid[-1][x]**2+f*(1-u_grid[-1][x]))+u_grid[-1][x]
            new_v_grid[-1][x] = delta_t * (D*((v_grid[-1][x+1]+v_grid[-1][x-1]+v_grid[-2][x]-4*v_grid[-1][x])/delta_x**2)+u_grid[-1][x]*v_grid[-1][x]**2-(f+k)*v_grid[-1][x])+v_grid[-1][x]

        for y in range(1, N - 1):
            new_u_grid[y][0] = delta_t * (D*((u_grid[y][1]+u_grid[y+1][0]+u_grid[y-1][0]-4*u_grid[y][0])/delta_x**2)-u_grid[y][0]*v_grid[y][0]**2+f*(1-u_grid[y][0]))+u_grid[y][0]
            new_v_grid[y][0] = delta_t * (D*((v_grid[y][1]+v_grid[y+1][0]+v_grid[y-1][0]-4*v_grid[y][0])/delta_x**2)+u_grid[y][0]*v_grid[y][0]**2-(f+k)*v_grid[y][0])+v_grid[y][0]
   
            new_u_grid[y][-1] = delta_t * (D*((u_grid[y][-2]+u_grid[y+1][-1]+u_grid[y-1][-1]-4*u_grid[y][-1])/delta_x**2)-u_grid[y][-1]*v_grid[y][-1]**2+f*(1-u_grid[y][-1]))+u_grid[y][-1]
            new_v_grid[y][-1] = delta_t * (D*((v_grid[y][-2]+v_grid[y+1][-1]+v_grid[y-1][-1]-4*v_grid[y][-1])/delta_x**2)+u_grid[y][-1]*v_grid[y][-1]**2-(f+k)*v_grid[y][-1])+v_grid[y][-1]
        
        
        for y in range(0, N):
            for x in range(0, N):
                new_u_grid[y][x] += 0.1 * random.random() - 0.05
                new_v_grid[y][x] += 0.1 * random.random() - 0.05

                if new_u_grid[y][x] < 0:
                    new_u_grid[y][x] = 0
                if new_v_grid[y][x] < 0:
                    new_v_grid[y][x] = 0
                if new_u_grid[y][x] > 1:
                    new_u_grid[y][x] = 1
                if new_v_grid[y][x] > 1:
                    new_v_grid[y][x] = 1


        u_grid = new_u_grid
        v_grid = new_v_grid

        if counter % 1 == 0:
            u_grid_list.append(u_grid)
            v_grid_list.append(v_grid)

    
    with open(output, 'wb') as fp:
        pickle.dump([u_grid_list,v_grid_list], fp)

    return u_grid_list, v_grid_list

if not run_again:
    with open(output, 'rb') as fp:
        u_grid_list, v_grid_list = pickle.load(fp)
else:
    u_grid_list, v_grid_list = calculate_diffusion()


times = [0, 1, 10, 100, 499]

for time in times:
    plt.imshow(u_grid_list[time])
    plt.title(f"t={time/1000}")
    plt.show()

for grid in u_grid_list:
    plt.imshow(grid)
    plt.draw()
    plt.pause(0.001)
    plt.clf()

for time in times:
    plt.imshow(v_grid_list[time])
    plt.title(f"t={time/1000}")
    plt.show()

for grid in v_grid_list:
    plt.imshow(grid)
    plt.draw()
    plt.pause(0.001)
    plt.clf()