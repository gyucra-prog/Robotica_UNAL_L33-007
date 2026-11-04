
# ==================================================
# PARTE 4 - EXPERIENCIA VISUAL EKF-SLAM
# Robot movil explorando un entorno desconocido
# ==================================================

import numpy as np
import matplotlib.pyplot as plt

from roboticstoolbox import (
    Bicycle,
    RandomPath,
    LandmarkMap,
    RangeBearingSensor,
    EKF
)

# ==================================================
# 1. CONFIGURACION DEL ENTORNO
# ==================================================

SEMILLA = 10
TIEMPO = 40

# ==================================================
# 2. COVARIANZA DE ODOMETRIA
# ==================================================

V = np.diag([
    0.02,
    np.deg2rad(0.5)
])**2

# ==================================================
# 3. CREAR ROBOT
# ==================================================

robot = Bicycle(
    covar=V,
    animation="car",
    workspace=10
)

# ==================================================
# 4. CREAR MAPA REAL
# ==================================================

mapa = LandmarkMap(
    20,
    workspace=10,
    seed=SEMILLA
)

# El robot no conoce este mapa.
# Solo el sensor lo utiliza para generar observaciones.

robot.control = RandomPath(
    workspace=mapa,
    seed=SEMILLA
)

# ==================================================
# 5. CONFIGURAR SENSOR
# ==================================================

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
    animate=True,
    seed=SEMILLA
)

# ==================================================
# 6. INCERTIDUMBRE INICIAL
# ==================================================

P0 = np.diag([
    0.05,
    0.05,
    np.deg2rad(0.5)
])**2

# ==================================================
# 7. CONFIGURAR EKF-SLAM
# ==================================================

slam = EKF(
    robot=(robot, V),
    P0=P0,
    sensor=(sensor, W),
    map=None
)

# ==================================================
# 8. EJECUTAR EXPLORACION ANIMADA
# ==================================================

print("Iniciando exploracion SLAM...")
print("El robot esta construyendo el mapa.")

slam.run(
    T=TIEMPO,
    animate=True
)

print("Exploracion finalizada.")

# ==================================================
# 9. MAPA REAL VS MAPA ESTIMADO
# ==================================================

fig, ax = plt.subplots(figsize=(11, 9))

# Mapa real
mapa.plot()

# Mapa construido por SLAM
slam.plot_map()

# Trayectoria real
robot.plot_xy(color='blue', label="Real")

# Trayectoria estimada
slam.plot_xy(color='red', label="Estimada")

# Elipses de confianza del robot
slam.plot_ellipse(N=15)

plt.title("EKF-SLAM: Mapa real y mapa estimado")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid(True)
plt.axis("equal")
plt.legend()
# ==================================================
# 10. GRAFICA DE INCERTIDUMBRE
# ==================================================

fig, ax = plt.subplots(figsize=(10, 5))

tiempo = slam.get_t()
incertidumbre = slam.get_Pnorm()

ax.plot(
    tiempo,
    incertidumbre,
    linewidth=2,
    color='purple'
)

ax.set_title("Evolucion de incertidumbre SLAM")
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Norma de covarianza")
ax.grid(True)

# ==================================================
# 11. RESULTADOS
# ==================================================

print("\n========== RESULTADOS SLAM ==========")

print("Pose estimada:")
print(slam.x_est[:3])

print("\nNumero de landmarks detectados:")
print(len(slam.landmarks))

print("\nLandmarks estimados:")
print(slam.get_map())

print("\nIncertidumbre final:")
print(slam.get_Pnorm()[-1])

print("\nDimensiones del estado:")
print(slam.x_est.shape)

# ==================================================
# 12. MOSTRAR GRAFICAS
# ==================================================

plt.show()
