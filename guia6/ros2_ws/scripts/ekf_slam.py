from roboticstoolbox import *
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# =================================
# 1. Covarianza de odometría
# =================================

V = np.diag([
    0.02,
    np.deg2rad(0.5)
])**2

# =================================
# 2. Robot
# =================================

robot = Bicycle(
    covar=V,
    animation=None
)

robot.control = RandomPath(workspace=10)

# =================================
# 3. Mapa real de landmarks
# =================================

mapa = LandmarkMap(20, workspace=10)

# =================================
# 4. Sensor de rango y orientación
# =================================

W = np.diag([
    0.1,
    np.deg2rad(1)
])**2

sensor = RangeBearingSensor(
    robot=robot,
    map=mapa,
    covar=W,
    angle=[-pi/2, pi/2],
    range=4
)

# =================================
# 5. Covarianza inicial
# =================================

P0 = np.diag([
    0.05,
    0.05,
    np.deg2rad(0.5)
])**2

# =================================
# 6. EKF-SLAM
# =================================

slam = EKF(
    robot=(robot, V),
    P0=P0,
    sensor=(sensor, W),
    map=None
)

# =================================
# 7. Ejecutar simulación
# =================================

slam.run(T=40)

# =================================
# 8. Graficar mapa y trayectorias
# =================================

fig, ax = plt.subplots(figsize=(10, 8))

# Mapa real de landmarks
mapa.plot()

# Trayectoria real
robot.plot_xy(color="blue")

# Trayectoria estimada por SLAM
slam.plot_xy(color="red")

# Elipses de confianza
slam.plot_ellipse()

ax.set_title("SLAM usando EKF")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.grid(True)
ax.axis("equal")

# Leyenda
ax.legend([
    "Landmarks reales",
    "Trayectoria real",
    "Trayectoria estimada"
])

# =================================
# 9. Evolución de incertidumbre
# =================================

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.plot(
    slam.get_t(),
    slam.get_Pnorm(),
    linewidth=2,
    label="Incertidumbre EKF-SLAM"
)

ax2.set_xlabel("Tiempo [s]")
ax2.set_ylabel("Incertidumbre")
ax2.set_title("Evolución de incertidumbre en SLAM")
ax2.grid(True)
ax2.legend()

plt.show()
