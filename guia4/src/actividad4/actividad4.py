import numpy as np
import matplotlib.pyplot as plt

# Trayectoria objetivo
path = np.array([
    [0, 0],
    [2, 4],
    [6, 5],
    [8, 10]
])

# Parámetros
lookahead = 0.5
Kd = 1.3

# Estado inicial
x = 0.0
y = 0.0
theta = 0.0

# Parámetros del modelo
dt = 0.1
L = 2.0
v = 0.5

# Punto actual de seguimiento
current_index = 0

# Iteraciones
iteraciones = 285

# Historial
X = []
Y = []


def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


for _ in range(iteraciones):

    # Distancia al punto actual
    dx = path[current_index][0] - x
    dy = path[current_index][1] - y

    dist = np.hypot(dx, dy)

    # Avanzar al siguiente punto si estamos suficientemente cerca
    if dist < lookahead and current_index < len(path) - 1:
        current_index += 1

    # Objetivo actual
    target = path[current_index]

    # Dirección hacia el objetivo
    dx = target[0] - x
    dy = target[1] - y

    theta_goal = np.arctan2(dy, dx)

    # Error angular normalizado
    error = angle_wrap(theta_goal - theta)

    # Control
    delta = Kd * error

    # Modelo cinemático
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += (v / L) * np.tan(delta) * dt

    # Guardar trayectoria
    X.append(x)
    Y.append(y)


# Gráfica
plt.plot(
    path[:, 0],
    path[:, 1],
    "--",
    label="Trayectoria objetivo"
)

plt.plot(
    X,
    Y,
    label="Robot"
)

plt.scatter(
    path[:, 0],
    path[:, 1],
    label="Puntos objetivo"
)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Seguimiento de trayectoria - Pure Pursuit")

plt.axis("equal")
plt.grid()
plt.legend()

plt.show()
