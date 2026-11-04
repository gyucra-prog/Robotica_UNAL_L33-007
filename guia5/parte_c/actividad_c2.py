import heapq
import numpy as np
import matplotlib.pyplot as plt

# Mapa: 0 = espacio libre, 1 = obstáculo
grid = np.zeros((25, 25), dtype=int)

# Obstáculo 1: rectángulo horizontal
grid[4:7, 3:11] = 1

# Obstáculo 2: rectángulo vertical
grid[8:17, 6:9] = 1

# Obstáculo 3: cuadrado
grid[3:8, 15:20] = 1

# Obstáculo 4: forma de L
grid[12:15, 12:21] = 1
grid[12:19, 12:15] = 1

# Obstáculo 5: barrera escalonada
grid[19:22, 3:8] = 1
grid[17:22, 6:9] = 1

# Obstáculo 6: rectángulo horizontal
grid[20:23, 13:22] = 1

# Inicio y objetivo
start = (1, 1)
goal = (23, 23)


# Heurística Manhattan
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Algoritmo A*
def astar(grid, start, goal):
    open_list = []
    heapq.heappush(
        open_list,
        (heuristic(start, goal), start)
    )

    came_from = {}
    cost = {start: 0}
    closed = set()

    while open_list:
        _, current = heapq.heappop(open_list)

        if current in closed:
            continue

        if current == goal:
            break

        closed.add(current)

        for dx, dy in [
            (-1, 0), (1, 0),
            (0, -1), (0, 1)
        ]:
            neighbor = (
                current[0] + dx,
                current[1] + dy
            )

            x, y = neighbor

            # Límites del mapa
            if not (
                0 <= x < grid.shape[1]
                and 0 <= y < grid.shape[0]
            ):
                continue

            # Evitar obstáculos
            if grid[y, x] == 1:
                continue

            new_cost = cost[current] + 1

            if (
                neighbor not in cost
                or new_cost < cost[neighbor]
            ):
                cost[neighbor] = new_cost

                priority = (
                    new_cost + heuristic(neighbor, goal)
                )

                heapq.heappush(
                    open_list,
                    (priority, neighbor)
                )

                came_from[neighbor] = current

    # Reconstruir la ruta
    if goal not in cost:
        return []

    path = []
    node = goal

    while node != start:
        path.append(node)
        node = came_from[node]

    path.append(start)
    path.reverse()

    return path


# Calcular trayectoria
path = astar(grid, start, goal)

# Visualizar
plt.figure(figsize=(9, 9))
plt.imshow(grid, origin="lower", cmap="gray_r")

if path:
    path = np.array(path)

    plt.plot(
        path[:, 0],
        path[:, 1],
        "r-",
        linewidth=2,
        label="Ruta A*"
    )

plt.scatter(
    *start,
    color="blue",
    s=90,
    label="Inicio"
)

plt.scatter(
    *goal,
    color="green",
    s=90,
    label="Objetivo"
)

plt.title("Planificación A* con obstáculos de distintas formas")
plt.xlabel("X")
plt.ylabel("Y")
plt.xticks(range(25))
plt.yticks(range(25))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Resultados
print("Inicio:", start)
print("Objetivo:", goal)
print("Objetivo alcanzado:", bool(path) and tuple(path[-1]) == goal)

if path:
    print("Número de movimientos:", len(path) - 1)
else:
    print("No se encontró una ruta.")
