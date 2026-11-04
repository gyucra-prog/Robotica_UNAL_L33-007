import numpy as np
import matplotlib.pyplot as plt

# Crear mapa: 0 = libre, 1 = obstáculo
grid = np.zeros((25, 25))

# Definir tres obstáculos
grid[4:9, 4:12] = 1
grid[11:16, 10:19] = 1
grid[18:22, 3:9] = 1

# Posición inicial y objetivo
start = np.array([1, 1])
goal = np.array([23, 23])
pos = start.copy()

path = [pos.copy()]


def is_free(p):
    x, y = p

    if not (0 <= x < 25 and 0 <= y < 25):
        return False

    return grid[y, x] == 0


# Simulación
for _ in range(1000):

    # Dirección hacia el objetivo
    direction = goal - pos

    # Avanzar por el eje con mayor diferencia
    if abs(direction[0]) > abs(direction[1]):
        step = np.array([np.sign(direction[0]), 0], dtype=int)
    else:
        step = np.array([0, np.sign(direction[1])], dtype=int)

    next_pos = pos + step

    if is_free(next_pos):
        pos = next_pos

    else:
        # Buscar una celda libre alrededor del robot
        moves = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1])
        ]

        for move in moves:
            candidate = pos + move

            if is_free(candidate):
                pos = candidate
                break

    path.append(pos.copy())

    if np.array_equal(pos, goal):
        break


# Convertir trayectoria
path = np.array(path)

# Visualizar mapa y trayectoria
plt.figure(figsize=(8, 8))
plt.imshow(grid, origin="lower", cmap="gray_r")

plt.plot(
    path[:, 0],
    path[:, 1],
    "g-",
    linewidth=2,
    label="Trayectoria"
)

plt.scatter(*start, color="blue", s=80, label="Inicio")
plt.scatter(*goal, color="red", s=80, label="Objetivo")

plt.title("Navegacion Bug2 con tres obstáculos")
plt.xlabel("X")
plt.ylabel("Y")
plt.xticks(range(25))
plt.yticks(range(25))
plt.grid(True)
plt.legend()
plt.show()

# Resultados
print("Posición final:", pos)
print("Objetivo:", goal)
print("Objetivo alcanzado:", np.array_equal(pos, goal))
print("Posiciones registradas:", len(path))
