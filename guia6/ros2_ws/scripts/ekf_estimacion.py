from roboticstoolbox import Bicycle, EKF, RandomPath
import numpy as np
import matplotlib.pyplot as plt

# Covarianza de odometría
V = np.diag([0.02, np.deg2rad(0.5)])**2

# Robot
robot = Bicycle(covar=V, animation=None)

# Trayectoria aleatoria
robot.control = RandomPath(workspace=10)

# Covarianza inicial
P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)])**2

# Filtro de Kalman Extendido
ekf = EKF(robot=(robot, V), P0=P0)

# Ejecutar simulación
ekf.run(T=20)

# Trayectoria real vs estimada
plt.figure(figsize=(8, 6))

robot.plot_xy(color='blue')
ekf.plot_xy(color='red')

plt.title('Trayectoria real vs estimada')
plt.grid(True)

# Incertidumbre
plt.figure(figsize=(8, 6))

plt.plot(
    ekf.get_t(),
    ekf.get_Pnorm(),
    linewidth=2
)

plt.xlabel('Tiempo [s]')
plt.ylabel('Incertidumbre')
plt.title('Crecimiento de incertidumbre')
plt.grid(True)

plt.show()
