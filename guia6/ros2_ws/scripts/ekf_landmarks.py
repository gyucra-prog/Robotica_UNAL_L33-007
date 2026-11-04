from roboticstoolbox import *
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Covarianza de odometría
V = np.diag([
    0.02,
    np.deg2rad(0.5)
])**2

# Robot
robot = Bicycle(
    covar=V,
    animation=None
)

robot.control = RandomPath(workspace=10)

# Mapa de landmarks
mapa = LandmarkMap(20, workspace=10)

# Covarianza del sensor
W = np.diag([
    0.1,
    np.deg2rad(1)
])**2

# Sensor de rango y orientación
sensor = RangeBearingSensor(
    robot=robot,
    map=mapa,
    covar=W,
    angle=[-pi/2, pi/2],
    range=4
)

# Covarianza inicial
P0 = np.diag([
    0.05,
    0.05,
    np.deg2rad(0.5)
])**2

# EKF con mapa conocido
ekf = EKF(
    robot=(robot, V),
    P0=P0,
    map=mapa,
    sensor=(sensor, W)
)

# Ejecutar simulación
ekf.run(T=40)
print("Landmarks:", mapa.landmarks.shape)
print("Trayectoria real:", robot.x.shape)
print("Tiempo EKF:", ekf.get_t().shape)
print("Incertidumbre:", ekf.get_Pnorm().shape)

# Graficar trayectorias
# Figura 1: trayectoria, landmarks y estimación EKF
fig, ax = plt.subplots(figsize=(8, 6))

mapa.plot()
robot.plot_xy(color="blue", label="Ideal")
ekf.plot_xy(color="red", label="Real")
ekf.plot_ellipse()

ax.set_title("EKF con landmarks")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.grid(True)
ax.axis("equal")
ax.legend()

# Figura 2: evolución de la incertidumbre
fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.plot(ekf.get_t(), ekf.get_Pnorm(), linewidth=2)
ax2.set_xlabel("Tiempo [s]")
ax2.set_ylabel("Incertidumbre")
ax2.set_title("Incertidumbre con correcciones")
ax2.grid(True)

plt.show()
