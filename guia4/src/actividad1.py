import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
L = 2.0
v = 1.0
delta = 0.3

# Estado inicial
x = 0.0
y = 0.0
theta = 0.0

# Paso de tiempo
dt = 0.1

# Historial de la trayectoria
X = []
Y = []

# Simulación
for _ in range(100):

    # Modelo cinemático
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = v / L * np.tan(delta)

    # Integración numérica
    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt

    # Guardar trayectoria
    X.append(x)
    Y.append(y)

# Graficar trayectoria
plt.plot(X, Y)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Trayectoria del robot")

plt.axis("equal")
plt.grid()

plt.show()
