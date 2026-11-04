import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import (
    Bicycle,
    EKF,
    LandmarkMap,
    RangeBearingSensor
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

# 3. DEFINIR LOS CUATRO PUNTOS
puntos = np.array([
    [2.0, 2.0],
    [8.0, 2.0],
    [8.0, 8.0],
    [2.0, 8.0]
])

# 4. CONTROLADOR DE TRAYECTORIA
indice = 0

def control_waypoints(robot, t, x):

    global indice

    # Posicion y orientacion
    px, py, theta = x[:3]

    # Punto objetivo
    objetivo = puntos[indice]

    dx = objetivo[0] - px
    dy = objetivo[1] - py

    distancia = np.hypot(dx, dy)

    # Comprobar llegada al waypoint
    if distancia < 0.3:

        if indice < len(puntos) - 1:
            indice += 1
        else:
            return np.array([0.0, 0.0])

        objetivo = puntos[indice]

        dx = objetivo[0] - px
        dy = objetivo[1] - py

        distancia = np.hypot(dx, dy)

    # Orientacion deseada
    theta_obj = np.arctan2(dy, dx)

    # Error angular
    error = theta_obj - theta

    error = np.arctan2(
        np.sin(error),
        np.cos(error)
    )

    # Control de direccion
    delta = np.clip(
        1.5 * error,
        -0.5,
        0.5
    )

    # Velocidad
    velocidad = min(1.0, distancia)

    return np.array([velocidad, delta])


robot.control = control_waypoints

# 5. CREAR MAPA DE LANDMARKS
mapa = LandmarkMap(
    20,
    workspace=10,
    seed=0
)

# 6. COVARIANZA DEL SENSOR
W = np.diag([
    0.1,
    np.deg2rad(1)
])**2

# 7. CONFIGURAR SENSOR RANGE-BEARING
sensor = RangeBearingSensor(
    robot=robot,
    map=mapa,
    covar=W,
    angle=[-np.pi/2, np.pi/2],
    range=4,
    seed=0
)

# 8. COVARIANZA INICIAL
P0 = np.diag([
    0.05,
    0.05,
    np.deg2rad(0.5)
])**2

# 9. CONFIGURAR EKF
ekf = EKF(
    robot=(robot, V),
    P0=P0,
    map=mapa,
    sensor=(sensor, W)
)

# 10. EJECUTAR SIMULACION
ekf.run(T=40)

# 11. GRAFICAR TRAYECTORIAS
plt.figure(figsize=(10, 8))

mapa.plot()

robot.plot_xy(color='blue')
ekf.plot_xy(color='red')

plt.scatter(
    puntos[:, 0],
    puntos[:, 1],
    color='green',
    marker='x',
    s=120,
    label='Waypoints'
)

for i, punto in enumerate(puntos):
    plt.annotate(
        f'P{i+1}',
        punto
    )

ekf.plot_ellipse()

plt.title('EKF con Landmarks')
plt.xlabel('Posicion X [m]')
plt.ylabel('Posicion Y [m]')
plt.grid(True)
plt.legend()

# 12. GRAFICAR INCERTIDUMBRE
plt.figure(figsize=(10, 5))

plt.plot(
    ekf.get_t(),
    ekf.get_Pnorm(),
    linewidth=2,
    color='purple'
)

plt.xlabel('Tiempo [s]')
plt.ylabel('Norma de covarianza')
plt.title('Incertidumbre del EKF con correcciones')
plt.grid(True)

# 13. RESULTADOS NUMERICOS
print("Pose estimada final:")
print(ekf.x_est)

print("\nCovarianza final:")
print(ekf.P_est)

print("\nIncertidumbre inicial:")
print(ekf.get_Pnorm()[0])

print("\nIncertidumbre final:")
print(ekf.get_Pnorm()[-1])
plt.show()
