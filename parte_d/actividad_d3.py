import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ==========================================
# 1. MAPA DE OCUPACIÓN
# ==========================================
SIZE = 40
grid = np.zeros((SIZE, SIZE), dtype=int)

# Obstáculos rectangulares
obstacles = [
    (3, 10, 3, 12),
    (14, 20, 0, 7),
    (7, 13, 16, 23),
    (16, 23, 11, 17),
    (25, 32, 4, 10),
    (32, 36, 12, 20),
    (3, 8, 27, 34),
    (12, 18, 25, 31),
    (20, 27, 22, 28),
    (28, 35, 25, 31),
    (5, 12, 36, 39),
    (17, 23, 34, 38),
    (24, 29, 14, 19),
    (33, 38, 32, 37),
    (0, 3, 15, 23),
    (19, 22, 2, 8),
    (35, 39, 2, 6),
    (1, 6, 20, 24),
]

# Cada tupla: (x_inicio, x_fin, y_inicio, y_fin)
for x1, x2, y1, y2 in obstacles:
    grid[y1:y2, x1:x2] = 1

start = (1, 1)
goal = (38, 38)

# Garantizar inicio y objetivo libres
grid[start[1], start[0]] = 0
grid[goal[1], goal[0]] = 0

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ==========================================
# 2. DISTANCE TRANSFORM MEDIANTE BFS
# ==========================================
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

# ==========================================
# 3. EXTRAER LA TRAYECTORIA
# ==========================================
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

    pos = min(candidates, key=lambda p: dist[p[1], p[0]])
    path.append(pos)

path = np.array(path)
reached = tuple(path[-1]) == goal

# ==========================================
# 4. RESULTADOS
# ==========================================
print("\n===== DISTANCE TRANSFORM =====")
print(f"Tamaño del mapa: {SIZE} x {SIZE}")
print(f"Inicio: {start}")
print(f"Objetivo: {goal}")
print(f"Celdas ocupadas: {np.sum(grid == 1)}")

if reached:
    print("Estado: objetivo alcanzado")
    print(f"Distancia mínima: {int(dist[start[1], start[0]])} movimientos")
    print(f"Movimientos calculados: {len(path) - 1}")
else:
    print("Estado: no se encontró una trayectoria completa")

# ==========================================
# 5. VISUALIZACIÓN
# ==========================================
fig, ax = plt.subplots(figsize=(12, 10))

distance_map = np.ma.masked_where(grid == 1, dist)

im = ax.imshow(
    distance_map,
    origin="lower",
    cmap="turbo",
    interpolation="nearest",
    extent=(-0.5, SIZE - 0.5, -0.5, SIZE - 0.5)
)

# Obstáculos
obs_y, obs_x = np.where(grid == 1)
ax.scatter(obs_x, obs_y, color="black", s=22, label="Obstáculos")

# Ruta calculada
if len(path) > 0:
    ax.plot(
        path[:, 0],
        path[:, 1],
        color="red",
        linewidth=2.5,
        marker="o",
        markersize=2.5,
        label="Trayectoria"
    )

# Inicio y objetivo
ax.scatter(*start, color="white", edgecolors="black",
           s=150, label="Inicio", zorder=5)

ax.scatter(*goal, color="lime", edgecolors="black",
           s=150, label="Objetivo", zorder=5)

ax.set_title("Distance Transform: navegación con obstáculos complejos")
ax.set_xlabel("Coordenada X")
ax.set_ylabel("Coordenada Y")
ax.set_xticks(range(SIZE))
ax.set_yticks(range(SIZE))
ax.grid(True, alpha=0.2)
ax.legend(loc="upper left")

fig.colorbar(im, ax=ax, label="Distancia mínima al objetivo")
plt.tight_layout()
plt.show()
