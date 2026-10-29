import numpy as np
import matplotlib.pyplot as plt

# Crear mapa: 0 = libre, 1 = obstáculo
grid = np.zeros((20, 20))
grid[8:12, 5:15] = 1

# Posición inicial y objetivo
start = np.array([2, 2])
goal = np.array([18, 18])
pos = start.copy()

path = [pos.copy()]


def is_free(p):
    x, y = p

    # Comprobar límites del mapa
    if not (0 <= x < 20 and 0 <= y < 20):
        return False

    return grid[y, x] == 0


# Simulación
for _ in range(500):

    # Dirección hacia el objetivo
    direction = goal - pos

    # Avanzar por el eje con mayor diferencia
    if abs(direction[0]) > abs(direction[1]):
        step = np.array([np.sign(direction[0]), 0], dtype=int)
    else:
        step = np.array([0, np.sign(direction[1])], dtype=int)

    next_pos = pos + step

    if is_free(next_pos):
        # Avanzar hacia el objetivo
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

# Convertir trayectoria a un arreglo
path = np.array(path)

# Visualizar resultados
plt.imshow(grid, origin="lower", cmap="gray_r")
plt.plot(path[:, 0], path[:, 1], "g-", label="Trayectoria")
plt.scatter(start[0], start[1], color="blue", label="Inicio")
plt.scatter(goal[0], goal[1], color="red", label="Objetivo")

plt.title("Navegación con Bug2")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.show()

# Mostrar resultado
print("Posición final:", pos)
print("Objetivo:", goal)
print("Objetivo alcanzado:", np.array_equal(pos, goal))
print("Número de posiciones registradas:", len(path))
