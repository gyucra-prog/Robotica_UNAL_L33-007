import heapq
import numpy as np
import matplotlib.pyplot as plt

# Crear mapa: 0 = libre, 1 = obstáculo
grid = np.zeros((20, 20))
grid[8:12, 5:15] = 1

# Posición inicial y objetivo (x, y)
start = (2, 2)
goal = (18, 18)


# Heurística: distancia Manhattan
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Algoritmo A*
def astar(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), start))

    came_from = {}
    cost = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            break

        # Explorar vecinos: izquierda, derecha, abajo, arriba
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            x, y = neighbor

            # Comprobar límites y obstáculos
            if not (0 <= x < grid.shape[1] and
                    0 <= y < grid.shape[0]):
                continue

            if grid[y, x] != 0:
                continue

            new_cost = cost[current] + 1

            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)

                heapq.heappush(open_list, (priority, neighbor))
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


# Calcular la ruta
path = astar(grid, start, goal)

if not path:
    print("No se encontró una ruta.")
else:
    path = np.array(path)

    # Visualizar mapa y trayectoria
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, origin="lower", cmap="gray_r")

    plt.plot(
        path[:, 0],
        path[:, 1],
        "r-",
        linewidth=2,
        label="Ruta A*"
    )

    plt.scatter(*start, color="blue", s=80, label="Inicio")
    plt.scatter(*goal, color="green", s=80, label="Objetivo")

    plt.title("Planificación de trayectoria con A*")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks(range(20))
    plt.yticks(range(20))
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Posición inicial:", start)
    print("Objetivo:", goal)
    print("Objetivo alcanzado:", tuple(path[-1]) == goal)
    print("Número de movimientos:", len(path) - 1)
