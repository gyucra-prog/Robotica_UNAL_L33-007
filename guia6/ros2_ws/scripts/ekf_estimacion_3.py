import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import Bicycle, EKF

# 1. CONFIGURACION
V = np.diag([
    0.02,
    np.deg2rad(0.5)
])**2

robot = Bicycle(
    covar=V,
    animation=None,
    workspace=10
)

# 2. DEFINIR LOS CUATRO PUNTOS
puntos = np.array([
    [2.0, 2.0],
    [8.0, 2.0],
    [8.0, 8.0],
    [2.0, 8.0]
])

# 3. CONTROLADOR DE WAYPOINTS
indice = 0

def control_waypoints(robot, t, x):

    global indice

    # Pose actual
    px = x[0]
    py = x[1]
    theta = x[2]

    # Punto objetivo
    objetivo = puntos[indice]

    dx = objetivo[0] - px
    dy = objetivo[1] - py

    distancia = np.hypot(dx, dy)

    # Comprobar si llego al punto
    if distancia < 0.3:

        if indice < len(puntos) - 1:
            indice += 1
        else:
            return np.array([0.0, 0.0])

        objetivo = puntos[indice]

        dx = objetivo[0] - px
        dy = objetivo[1] - py

    # Orientacion deseada
    theta_obj = np.arctan2(dy, dx)

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
    v = 1.0

    return np.array([v, delta])


robot.control = control_waypoints

# 4. COVARIANZA INICIAL
P0 = np.diag([
    0.05,
    0.05,
    np.deg2rad(0.5)
])**2

# 5. CREAR EKF
ekf = EKF(
    robot=(robot, V),
    P0=P0
)

# 6. EJECUTAR SIMULACION
ekf.run(T=40)

# 7. GRAFICAR TRAYECTORIAS
plt.figure(figsize=(10, 8))

robot.plot_xy(color='blue', label="Ideal")
ekf.plot_xy(color='red', label="Real")

plt.scatter(
    puntos[:, 0],
    puntos[:, 1],
    color='green',
    marker='x',
    s=120,
    label='Puntos objetivo'
)

for i, punto in enumerate(puntos):
    plt.annotate(
        f'P{i+1}',
        punto
    )

plt.title('EKF - Trayectoria con cuatro puntos')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.grid()
plt.legend()

# 8. INCERTIDUMBRE
plt.figure(figsize=(10, 5))

plt.plot(
    ekf.get_t(),
    ekf.get_Pnorm(),
    linewidth=2
)

plt.title('Incertidumbre del EKF')
plt.xlabel('Tiempo [s]')
plt.ylabel('Norma de covarianza')
plt.grid()

plt.show()
