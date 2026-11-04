from roboticstoolbox import Bicycle, EKF, RandomPath
import numpy as np
import matplotlib.pyplot as plt

# 2. DEFINIR RUIDO DE ODOMETRIA
V = np.diag([
    0.02,
    np.deg2rad(0.5)
])**2

# 3. CREAR ROBOT MOVIL
robot = Bicycle(
    covar=V,
    animation=None,
    workspace=10
)

# Generar trayectoria aleatoria reproducible
robot.control = RandomPath(
    workspace=robot,
    seed=0
)

# 4. DEFINIR INCERTIDUMBRE INICIAL
P0 = np.diag([
    0.05,
    0.05,
    np.deg2rad(0.5)
])**2

# 5. CREAR FILTRO DE KALMAN EXTENDIDO
ekf = EKF(
    robot=(robot, V),
    P0=P0
)

# 6. EJECUTAR SIMULACION
ekf.run(T=20)

# 7. GRAFICAR TRAYECTORIAS
plt.figure(figsize=(9, 7))

robot.plot_xy(color='blue')
ekf.plot_xy(color='red')

plt.title('Trayectoria real vs estimada')
plt.xlabel('Posicion X [m]')
plt.ylabel('Posicion Y [m]')
plt.grid(True)

# 8. GRAFICAR INCERTIDUMBRE
t = ekf.get_t()
incertidumbre = ekf.get_Pnorm()

plt.figure(figsize=(9, 5))

plt.plot(
    t,
    incertidumbre,
    linewidth=2,
    color='purple'
)

plt.xlabel('Tiempo [s]')
plt.ylabel('Norma de covarianza')
plt.title('Evolucion de la incertidumbre del EKF')
plt.grid(True)

# 9. MOSTRAR RESULTADOS NUMERICOS
print("Pose estimada final:")
print(ekf.x_est)

print("\nCovarianza estimada final:")
print(ekf.P_est)

print("\nIncertidumbre inicial:", incertidumbre[0])
print("Incertidumbre final:", incertidumbre[-1])
plt.show()
