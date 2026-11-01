import numpy as np
import matplotlib.pyplot as plt

# Estado inicial
x = 0.0
y = 0.0
theta = 0.0

# Objetivo
goal = np.array([5.0, 5.0])

# Ganancias del controlador
Kv = 0.5
Kh = 1.5

# Parámetros del modelo
dt = 0.1
L = 2.0

# Historial
X = []
Y = []

# Distancia inicial
distance = 10.0


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


while distance > 0.1:

    # Error de posición
    dx = goal[0] - x
    dy = goal[1] - y

    # Distancia al objetivo
    distance = np.sqrt(dx**2 + dy**2)

    # Dirección hacia el objetivo
    theta_goal = np.arctan2(dy, dx)

    # Error angular
    error_theta = wrap_angle(theta_goal - theta)

    # Control proporcional
    v = Kv * distance
    delta = Kh * error_theta

    # Modelo cinemático tipo bicicleta
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += (v / L) * np.tan(delta) * dt

    # Guardar trayectoria
    X.append(x)
    Y.append(y)

# Mostrar resultado
plt.plot(X, Y, label="Trayectoria del robot")
plt.scatter(goal[0], goal[1], label="Objetivo")

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Control: ir a un punto")

plt.axis("equal")
plt.grid()
plt.legend()

plt.show()
