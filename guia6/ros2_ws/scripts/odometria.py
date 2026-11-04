from roboticstoolbox import Bicycle
import numpy as np
import matplotlib.pyplot as plt

# 1. Covarianza del ruido
V = np.diag([0.2, np.deg2rad(0.5)])**2

print("Matriz de covarianza V:")
print(V)

# 2. Modelos de los robots
# Robot con ruido
robot = Bicycle(covar=V)

# Robot ideal, sin ruido
robot_ideal = Bicycle(covar=np.zeros((2, 2)))

# 3. Inicializar las trayectorias
x_real = []
y_real = []

x_ideal = []
y_ideal = []

# Pose inicial: [x, y, theta]
qp_real = [0, 0, 0]
qp_ideal = [0, 0, 0]

# 4. Simulación
for _ in range(200):

    # Movimiento del robot con ruido
    odo_real = robot.step((1.0, 0.2))
    qp_real = robot.f(qp_real, odo_real)

    # Movimiento del robot ideal
    odo_ideal = robot_ideal.step((1.0, 0.2))
    qp_ideal = robot_ideal.f(qp_ideal, odo_ideal)

    # Almacenar trayectoria con ruido
    x_real.append(qp_real[0])
    y_real.append(qp_real[1])

    # Almacenar trayectoria ideal
    x_ideal.append(qp_ideal[0])
    y_ideal.append(qp_ideal[1])

# 5. Graficar trayectorias
plt.figure(figsize=(8, 6))

plt.plot(
    x_ideal,
    y_ideal,
    label="Trayectoria estimada sin ruido",
    linewidth=1
)

plt.plot(
    x_real,
    y_real,
    label="Trayectoria estimada con ruido",
    linewidth=1
)

plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.title("Comparación de trayectorias")

plt.grid(True)
plt.axis("equal")
plt.legend()

plt.show()
