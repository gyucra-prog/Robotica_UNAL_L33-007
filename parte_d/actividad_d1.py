import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# =========================
# MAPA DE OCUPACIÓN
# =========================
SIZE = 20

grid = np.zeros((SIZE, SIZE), dtype=int)
grid[8:12, 5:15] = 1

start = (18, 18)
goal = (2, 2)

# =========================
# DISTANCE TRANSFORM
# =========================
dist = np.full((SIZE, SIZE), np.inf)

dist[goal[1], goal[0]] = 0
queue = deque([goal])

# Calcular las distancias mediante BFS
while queue:
    x, y = queue.popleft()

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy

        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            continue

        if grid[ny, nx] == 1:
            continue

        if np.isinf(dist[ny, nx]):
            dist[ny, nx] = dist[y, x] + 1
            queue.append((nx, ny))

# =========================
# EXTRAER LA TRAYECTORIA
# =========================
path = [start]
pos = start

for _ in range(SIZE * SIZE):
    if pos == goal:
        break

    x, y = pos
    neighbors = []

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy

        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            continue

        if grid[ny, nx] == 1:
            continue

        if dist[ny, nx] < dist[y, x]:
            neighbors.append((nx, ny))

    if not neighbors:
        break

    # Elegir el vecino con menor distancia al objetivo
    pos = min(neighbors, key=lambda p: dist[p[1], p[0]])
    path.append(pos)

path = np.array(path)

# =========================
# RESULTADOS
# =========================
if tuple(path[-1]) == goal:
    print("Objetivo alcanzado.")
    print(f"Distancia óptima: {int(dist[start[1], start[0]])} movimientos")
    print(f"Movimientos de la trayectoria: {len(path) - 1}")
else:
    print("No se pudo alcanzar el objetivo.")

# =========================
# VISUALIZACIÓN
# =========================
plt.figure(figsize=(9, 7))

distance_map = np.ma.masked_where(grid == 1, dist)

plt.imshow(
    distance_map,
    origin="lower",
    cmap="hot",
    interpolation="nearest"
)

obs_y, obs_x = np.where(grid == 1)
plt.scatter(obs_x, obs_y, c="black", s=25, label="Obstáculos")

if len(path) > 0:
    plt.plot(
        path[:, 0],
        path[:, 1],
        "cyan",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Trayectoria"
    )

plt.scatter(*start, c="blue", s=100, label="Inicio")
plt.scatter(*goal, c="lime", edgecolors="black", s=100, label="Objetivo")

plt.title("Distance Transform - Mapa de distancias")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Distancia al objetivo")
plt.legend()
plt.grid(True)
plt.xticks(range(SIZE))
plt.yticks(range(SIZE))
plt.show()
