import numpy as np
import matplotlib.pyplot as plt

# Campo de atracción del objetivo
def sensorfield(x, y):
    xc, yc = 60, 90
    return 1000 / ((x - xc)**2 + (y - yc)**2 + 50)

# Parámetros
dt = 0.1
Ks = 17
Kv = 14
d = 2

# Estado inicial del robot
x, y, theta = 10, 10, 0

traj_x, traj_y = [], []

# Simulación
for i in range(222):

    # Lecturas de los sensores
    s_left = sensorfield(
        x - d * np.sin(theta),
        y + d * np.cos(theta)
    )

    s_right = sensorfield(
        x + d * np.sin(theta),
        y - d * np.cos(theta)
    )

    # Ley de control
    v = -Kv * (s_left + s_right)
    omega = Ks * (s_right - s_left)

    # Actualización de la posición
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += omega * dt

    # Guardar trayectoria
    traj_x.append(x)
    traj_y.append(y)

# Visualización
plt.plot(traj_x, traj_y, label="Trayectoria")
plt.scatter(60, 90, color="red", label="Objetivo")
plt.scatter(10, 10, color="blue", label="Inicio")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Navegación reactiva - Braitenberg")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
