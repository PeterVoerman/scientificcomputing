import numpy as np
import matplotlib.pyplot as plt

length = 1
intervals = 1000
t_end = 1
c = 1

delta_x = length / intervals
delta_t = 0.001

constant = c ** 2 * delta_t ** 2 / delta_x ** 2

string_pos = np.zeros(intervals)
for x in range(intervals):
    string_pos[x] = np.sin(5 * np.pi * x * delta_x)

#string_pos = [np.sin(5 * np.pi * x) for x in np.arange(0, length + delta_x, delta_x)]
old_pos = string_pos

x_list = np.arange(0, length, delta_x)

counter = 0
for t in np.arange(0, t_end, delta_t):
    print(f"{t:.2f}/{t_end}", end='\r')

    counter += 1

    new_pos = np.zeros(len(string_pos))
    new_pos[0] = string_pos[0]
    new_pos[-1] = string_pos[-1]

    for i in range(1, len(string_pos) - 1):
        new_pos[i] = constant * (string_pos[i + 1] + string_pos[i - 1] - 2 * string_pos[i]) - old_pos[i] + 2 * string_pos[i]

    old_pos = string_pos
    string_pos = new_pos

    if counter % 1 == 0:
        plt.plot(x_list, string_pos)
        plt.xlim(0, length)
        plt.ylim(-1, 1)
        plt.draw()
        plt.pause(0.001)
        plt.clf()

plt.plot(x_list, string_pos)
plt.xlim(0, length)
plt.ylim(-1, 1)
plt.show()

