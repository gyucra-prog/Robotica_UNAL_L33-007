import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import (
    Bicycle,
    RandomPath,
    LandmarkMap,
    RangeBearingSensor,
    EKF
)

# 1. COVARIANZA DE ODOMETRIA
V = np.diag([
    0.02,
    np.deg2rad(0.5)
])**2

# 2. CREAR ROBOT
robot = Bicycle(
    covar=V,
    animation=None,
    workspace=10
)

robot.control = RandomPath(
    workspace=10,
    seed=0
)

# 3. CREAR MAPA REAL
mapa = LandmarkMap(
    20,
    workspace=10,
    seed=0
)

# 4. CONFIGURAR SENSOR
W = np.diag([
    0.1,
    np.deg2rad(1)
])**2

sensor = RangeBearingSensor(
    robot=robot,
    map=mapa,
    covar=W,
    angle=[-np.pi/2, np.pi/2],
    range=4,
    seed=0
)

# 5. COVARIANZA INICIAL
P0 = np.diag([
    0.05,
    0.05,
    np.deg2rad(0.5)
])**2

# 6. CONFIGURAR EKF-SLAM
slam = EKF(
    robot=(robot, V),
    P0=P0,
    sensor=(sensor, W),
    map=None,
    animate=False
)

# 7. EJECUTAR SIMULACION
slam.run(T=40)

# 8. GRAFICAR MAPA Y TRAYECTORIAS
plt.figure(figsize=(11, 8))

# Mapa real
mapa.plot()

# Mapa estimado por SLAM
slam.plot_map()

# Trayectoria real
robot.plot_xy(color='blue', label="Real")

# Trayectoria estimada
slam.plot_xy(color='red', label="Estimada")

# Elipses de confianza
slam.plot_ellipse()

plt.title('EKF-SLAM: Localizacion y Mapeo Simultaneo')
plt.xlabel('Posicion X [m]')
plt.ylabel('Posicion Y [m]')
plt.grid(True)
plt.axis('equal')
plt.legend()

# 9. GRAFICAR INCERTIDUMBRE
plt.figure(figsize=(10, 6))

plt.plot(
    slam.get_t(),
    slam.get_Pnorm(),
    linewidth=2,
    color='purple'
)

plt.xlabel('Tiempo [s]')
plt.ylabel('Norma de covarianza')
plt.title('Evolucion de incertidumbre en EKF-SLAM')
plt.grid(True)

# 10. RESULTADOS NUMERICOS
print("Pose estimada del robot:")
print(slam.x_est[:3])

print("\nEstado completo estimado:")
print(slam.x_est)

print("\nDimensiones del estado:")
print(slam.x_est.shape)

print("\nCovarianza final:")
print(slam.P_est)

print("\nIncertidumbre inicial:")
print(slam.get_Pnorm()[0])

print("\nIncertidumbre final:")
print(slam.get_Pnorm()[-1])
plt.show()
