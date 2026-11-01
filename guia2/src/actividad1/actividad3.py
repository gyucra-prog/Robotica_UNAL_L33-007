import numpy as np
import matplotlib.pyplot as plt

# Línea: ax + by + c = 0
a = 1
b = -2
c = 4

# Pose inicial
x = 8.0
y = 5.0
theta = np.pi / 2

# Parámetros del controlador
Kd = 0.5
Kh = 1.0

# Parámetros del modelo
dt = 0.1
L = 2.0
v = 1.0

# Número de iteraciones
iteraciones = 200

# Historial
X = []
Y = []


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


for _ in range(iteraciones):

    # Distancia firmada a la línea
    distance = (a * x + b * y + c) / np.sqrt(a**2 + b**2)

    # Orientación de la línea
    theta_line = np.arctan2(a, -b)

    # Error angular
    error_theta = wrap_angle(theta_line - theta)

    # Control
    delta = Kd * distance + Kh * error_theta

    # Modelo cinemático
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += (v / L) * np.tan(delta) * dt

    # Guardar trayectoria
    X.append(x)
    Y.append(y)


# Crear la línea objetivo
x_line = np.linspace(3, 25, 100)

if b != 0:
    y_line = (-a * x_line - c) / b
else:
    x_line = np.full(100, -c / a)
    y_line = np.linspace(-5, 10, 100)


# Gráfica
plt.plot(X, Y, label="Trayectoria del robot")
plt.plot(x_line, y_line, "--", label="Línea objetivo")

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Seguimiento de línea")

plt.axis("equal")
plt.grid()
plt.legend()

plt.show()
