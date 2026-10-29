import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# 1. MAPA DE OCUPACIÓN
SIZE = 30
grid = np.zeros((SIZE, SIZE), dtype=int)

# Obstáculos de distintas formas y tamaños
grid[3:10, 3:6] = 1
grid[2:5, 9:17] = 1
grid[7:18, 8:11] = 1
grid[6:13, 14:21] = 1
grid[15:25, 4:7] = 1
grid[17:20, 10:18] = 1
grid[21:27, 12:15] = 1
grid[16:19, 21:28] = 1
grid[22:27, 22:25] = 1
grid[25:28, 5:11] = 1

start = (1, 1)
goal = (28, 28)

# Asegurar que inicio y objetivo estén libres
grid[start[1], start[0]] = 0
grid[goal[1], goal[0]] = 0

# Movimientos permitidos: arriba, abajo, izquierda, derecha
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 2. DISTANCE TRANSFORM MEDIANTE BFS
dist = np.full((SIZE, SIZE), np.inf)
dist[goal[1], goal[0]] = 0

queue = deque([goal])

while queue:
    x, y = queue.popleft()

    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy

        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            continue

        if grid[ny, nx] == 1:
            continue

        if np.isinf(dist[ny, nx]):
            dist[ny, nx] = dist[y, x] + 1
            queue.append((nx, ny))

# 3. EXTRAER LA RUTA ÓPTIMA
path = [start]
pos = start

for _ in range(SIZE * SIZE):
    if pos == goal:
        break

    x, y = pos
    candidates = []

    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy

        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            continue

        if grid[ny, nx] == 1:
            continue

        if dist[ny, nx] < dist[y, x]:
            candidates.append((nx, ny))

    if not candidates:
        break

    # Escoger la celda libre con menor distancia restante
    pos = min(candidates, key=lambda p: dist[p[1], p[0]])
    path.append(pos)

path = np.array(path)

# 4. RESULTADOS
reached = tuple(path[-1]) == goal

print("\n===== RESULTADOS DISTANCE TRANSFORM =====")
print(f"Mapa: {SIZE} x {SIZE}")
print(f"Inicio: {start}")
print(f"Objetivo: {goal}")
print(f"Obstáculos: {np.sum(grid == 1)} celdas ocupadas")

if reached:
    print("Estado: objetivo alcanzado")
    print(f"Distancia mínima: {int(dist[start[1], start[0]])} movimientos")
    print(f"Movimientos de la ruta: {len(path) - 1}")
else:
    print("Estado: no se encontró una ruta al objetivo")

# 5. VISUALIZACIÓN
fig, ax = plt.subplots(figsize=(11, 9))

# Mapa de distancias; los obstáculos quedan enmascarados
distance_map = np.ma.masked_where(grid == 1, dist)

im = ax.imshow(
    distance_map,
    origin="lower",
    cmap="viridis",
    interpolation="nearest",
    extent=(-0.5, SIZE - 0.5, -0.5, SIZE - 0.5)
)

# Obstáculos
obs_y, obs_x = np.where(grid == 1)
ax.scatter(obs_x, obs_y, c="black", s=25, label="Obstáculos")

# Ruta calculada
if len(path) > 0:
    ax.plot(
        path[:, 0],
        path[:, 1],
        color="red",
        linewidth=2.5,
        marker="o",
        markersize=3,
        label="Ruta óptima"
    )

# Inicio y objetivo
ax.scatter(*start, c="cyan", edgecolors="black", s=150, label="Inicio", zorder=5)
ax.scatter(*goal, c="yellow", edgecolors="black", s=150, label="Objetivo", zorder=5)

ax.set_title("Distance Transform: navegación entre múltiples obstáculos")
ax.set_xlabel("Coordenada X")
ax.set_ylabel("Coordenada Y")
ax.set_xticks(range(SIZE))
ax.set_yticks(range(SIZE))
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left")
fig.colorbar(im, ax=ax, label="Distancia mínima al objetivo")

plt.tight_layout()
plt.show()
