import heapq
import numpy as np
import matplotlib.pyplot as plt

# 1. CREAR EL MAPA
SIZE = 30
grid = np.zeros((SIZE, SIZE), dtype=int)

# Barreras que forman pasillos
grid[4:25, 5:7] = 1
grid[0:21, 11:13] = 1
grid[7:30, 17:19] = 1
grid[0:23, 23:25] = 1

# Aberturas en las barreras
grid[18:22, 5:7] = 0
grid[5:9, 11:13] = 0
grid[22:26, 17:19] = 0
grid[4:8, 23:25] = 0

# Inicio y objetivo
start = (1, 15)
goal = (28, 15)

# 2. HEURÍSTICA MANHATTAN
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 3. ALGORITMO A*
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

            if not (0 <= x < SIZE and 0 <= y < SIZE):
                continue

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

    # Reconstrucción de la ruta
    if goal not in cost:
        return [], 0

    path = []
    node = goal

    while node != start:
        path.append(node)
        node = came_from[node]

    path.append(start)
    path.reverse()

    return path, cost[goal]

# 4. CALCULAR LA RUTA
path, total_cost = astar(grid, start, goal)

# 5. VISUALIZAR RESULTADOS
plt.figure(figsize=(11, 9))

plt.imshow(
    grid,
    origin="lower",
    cmap="gray_r",
    extent=(-0.5, SIZE - 0.5, -0.5, SIZE - 0.5)
)

# Ruta directa de referencia, no necesariamente transitable
plt.plot(
    [start[0], goal[0]],
    [start[1], goal[1]],
    "b--",
    linewidth=2,
    label="Referencia directa"
)

if path:
    path_array = np.array(path)

    plt.plot(
        path_array[:, 0],
        path_array[:, 1],
        "r-",
        linewidth=2.5,
        label="Ruta calculada por A*"
    )

plt.scatter(
    *start,
    color="lime",
    edgecolors="black",
    s=120,
    label="Inicio",
    zorder=3
)

plt.scatter(
    *goal,
    color="orange",
    edgecolors="black",
    s=120,
    label="Objetivo",
    zorder=3
)

plt.title("A*: búsqueda de una ruta de coste mínimo")
plt.xlabel("X")
plt.ylabel("Y")
plt.xticks(range(SIZE))
plt.yticks(range(SIZE))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 6. RESULTADOS
if path:
    print("Ruta encontrada: sí")
    print("Inicio:", start)
    print("Objetivo:", goal)
    print("Coste total:", total_cost)
    print("Número de movimientos:", len(path) - 1)
else:
    print("No existe una ruta hasta el objetivo.")

