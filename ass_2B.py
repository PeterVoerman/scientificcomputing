N = 100

grid = np.zeros((N,N))

cluster = [[50, N-1]]

while len(cluster) < 100:
	while True:
		location_x = np.random.randint(0, 100)
		location_y = 0
		new_location_y = location_y
		new_location_x = location_x

		direction = np.random.choice(["e", "n", "w", "s"])

		if direction == "e":
			new_location_x += 1
		if direction == "w":
			new_location_x -= 1
		if direction == "n":
			new_location_y += 1
		if direction == "s":
			new_location_y -= 1

		