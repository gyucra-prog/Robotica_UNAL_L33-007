import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from roboticstoolbox import (
    Bicycle,
    RandomPath,
    LandmarkMap,
    RangeBearingSensor,
    EKF
)

# ==========================================================
# 1. CONFIGURACION
# ==========================================================

SEMILLA = 15
TIEMPO = 40
NUM_LANDMARKS = 20

# Ruido del movimiento: velocidad y direccion
V = np.diag([
    0.05,
    np.deg2rad(1.5)
])**2

# Ruido del sensor: distancia y angulo
W = np.diag([
    0.20,
    np.deg2rad(2.5)
])**2

# Incertidumbre inicial
P0 = np.diag([
    0.10,
    0.10,
    np.deg2rad(2)
])**2


# ==========================================================
# 2. CREAR CADA SIMULACION
# ==========================================================

def crear_simulacion(modo):

    np.random.seed(SEMILLA)

    # Robot
    robot = Bicycle(
        covar=V,
        animation=None,
        workspace=10
    )

    # Trayectoria
    robot.control = RandomPath(
        workspace=10,
        seed=SEMILLA
    )

    # Mapa real
    mapa = LandmarkMap(
        NUM_LANDMARKS,
        workspace=10,
        seed=SEMILLA
    )

    # Sensor de distancia y orientacion
    sensor = RangeBearingSensor(
        robot=robot,
        map=mapa,
        covar=W,
        range=5,
        angle=[-np.pi/2, np.pi/2],
        seed=SEMILLA
    )

    # ------------------------------------------------------
    # MODO 1: SOLO ODOMETRIA
    # ------------------------------------------------------

    if modo == "odometria":

        ekf = EKF(
            robot=(robot, V),
            P0=P0
        )

    # ------------------------------------------------------
    # MODO 2: EKF CON MAPA CONOCIDO
    # ------------------------------------------------------

    elif modo == "mapa":

        ekf = EKF(
            robot=(robot, V),
            P0=P0,
            map=mapa,
            sensor=(sensor, W)
        )

    # ------------------------------------------------------
    # MODO 3: EKF-SLAM
    # ------------------------------------------------------

    elif modo == "slam":

        ekf = EKF(
            robot=(robot, V),
            P0=P0,
            sensor=(sensor, W),
            map=None
        )

    # Ejecutar simulacion
    ekf.run(T=TIEMPO, animate=False)

    return robot, mapa, ekf


# ==========================================================
# 3. EJECUTAR LOS TRES METODOS
# ==========================================================

print("Simulando solo odometria...")
robot_odo, mapa_odo, ekf_odo = crear_simulacion("odometria")

print("Simulando EKF con mapa conocido...")
robot_ekf, mapa_ekf, ekf_ekf = crear_simulacion("mapa")

print("Simulando EKF-SLAM...")
robot_slam, mapa_slam, ekf_slam = crear_simulacion("slam")

print("Simulaciones finalizadas.")


# ==========================================================
# 4. EXTRAER INFORMACION
# ==========================================================

def extraer_datos(robot, ekf):

    # Trayectoria real
    real = np.asarray(robot.x_hist)

    # Trayectoria estimada
    estimada = np.asarray(ekf.get_xyt())

    # Asegurar igual cantidad de muestras
    n = min(len(real), len(estimada))

    real = real[:n, :3]
    estimada = estimada[:n, :3]

    # Error de posicion
    error = np.linalg.norm(
        real[:, :2] - estimada[:, :2],
        axis=1
    )

    tiempo = ekf.get_t()[:n]

    return real, estimada, error, tiempo


datos_odo = extraer_datos(robot_odo, ekf_odo)
datos_ekf = extraer_datos(robot_ekf, ekf_ekf)
datos_slam = extraer_datos(robot_slam, ekf_slam)

N = min(
    len(datos_odo[0]),
    len(datos_ekf[0]),
    len(datos_slam[0])
)

# Recortar para que las tres animaciones avancen juntas
datos = [tuple(x[:N] for x in d) for d in
         [datos_odo, datos_ekf, datos_slam]]


# ==========================================================
# 5. OBTENER LOS LANDMARKS REALES
# ==========================================================

# LandmarkMap.landmarks tiene dimensiones 2 x N
landmarks_reales = mapa_slam.landmarks.T

# Coordenadas estimadas por SLAM
# Se extraen del estado del filtro en cada instante.
landmarks_estimados = []

for h in ekf_slam.history[:N]:

    estado = np.asarray(h.xest)

    # Las tres primeras variables corresponden al robot.
    mapa_estimado = estado[3:].reshape(-1, 2)

    landmarks_estimados.append(mapa_estimado)


# ==========================================================
# 6. CONFIGURAR LA FIGURA
# ==========================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 6)
)

fig.suptitle(
    "Comparacion de localizacion: Odometria, EKF y SLAM",
    fontsize=15
)

titulos = [
    "1. Solo odometria",
    "2. EKF con mapa conocido",
    "3. EKF-SLAM"
]

colores = [
    "tab:blue",
    "tab:green",
    "tab:red"
]

lineas_reales = []
lineas_estimadas = []
robots = []
puntos_slam = []

for i, ax in enumerate(axes):

    ax.set_title(titulos[i])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_aspect("equal")
    ax.grid(True)

    # Landmarks reales
    ax.scatter(
        landmarks_reales[:, 0],
        landmarks_reales[:, 1],
        marker="x",
        color="black",
        s=65,
        label="Landmarks reales"
    )

    # Trayectoria real
    lr, = ax.plot(
        [],
        [],
        "k--",
        linewidth=1.5,
        label="Trayectoria real"
    )

    # Trayectoria estimada
    le, = ax.plot(
        [],
        [],
        color=colores[i],
        linewidth=2,
        label="Trayectoria estimada"
    )

    # Robot
    rb, = ax.plot(
        [],
        [],
        "o",
        color="orange",
        markersize=8,
        label="Robot"
    )

    # Puntos estimados por SLAM
    if i == 2:
        ps = ax.scatter(
            [],
            [],
            marker="o",
            facecolors="none",
            edgecolors="red",
            s=75,
            label="Landmarks estimados"
        )
        puntos_slam.append(ps)

    lineas_reales.append(lr)
    lineas_estimadas.append(le)
    robots.append(rb)

    ax.legend(loc="upper right", fontsize=8)


# ==========================================================
# 7. PANEL DE ERROR
# ==========================================================

fig_error, ax_error = plt.subplots(figsize=(10, 5))

ax_error.set_title("Comparacion del error de posicion")
ax_error.set_xlabel("Tiempo (s)")
ax_error.set_ylabel("Error (m)")
ax_error.grid(True)

error_odo, = ax_error.plot(
    [], [], label="Solo odometria"
)

error_ekf, = ax_error.plot(
    [], [], label="EKF mapa conocido"
)

error_slam, = ax_error.plot(
    [], [], label="EKF-SLAM"
)

ax_error.legend()

tiempo = datos[0][3]

ax_error.set_xlim(tiempo[0], tiempo[-1])

max_error = max(np.max(d[2]) for d in datos)
ax_error.set_ylim(0, max(0.1, max_error * 1.2))


# ==========================================================
# 8. ANIMACION
# ==========================================================

def actualizar(frame):

    for i, ax in enumerate(axes):

        real, estimada, error, t = datos[i]

        # Trayectoria real
        lineas_reales[i].set_data(
            real[:frame+1, 0],
            real[:frame+1, 1]
        )

        # Trayectoria estimada
        lineas_estimadas[i].set_data(
            estimada[:frame+1, 0],
            estimada[:frame+1, 1]
        )

        # Posicion del robot
        robots[i].set_data(
            [estimada[frame, 0]],
            [estimada[frame, 1]]
        )

    # Actualizar landmarks estimados por SLAM
    mapa_actual = landmarks_estimados[frame]

    if len(mapa_actual) > 0:

        puntos_slam[0].set_offsets(mapa_actual)

    else:

        puntos_slam[0].set_offsets(
            np.empty((0, 2))
        )

    # Actualizar errores
    error_odo.set_data(
        tiempo[:frame+1],
        datos[0][2][:frame+1]
    )

    error_ekf.set_data(
        tiempo[:frame+1],
        datos[1][2][:frame+1]
    )

    error_slam.set_data(
        tiempo[:frame+1],
        datos[2][2][:frame+1]
    )

    fig.suptitle(
        f"Comparacion de metodos | Tiempo: {tiempo[frame]:.2f} s"
    )

    return []


animacion = FuncAnimation(
    fig,
    actualizar,
    frames=N,
    interval=50,
    repeat=False,
    blit=False
)

plt.tight_layout()
plt.show()


# ==========================================================
# 9. RESULTADOS
# ==========================================================

nombres = [
    "Solo odometria",
    "EKF con mapa conocido",
    "EKF-SLAM"
]

print("\n========== RESULTADOS ==========")

for nombre, d in zip(nombres, datos):

    error = d[2]

    print(f"\n{nombre}")
    print(f"Error medio: {np.mean(error):.3f} m")
    print(f"Error maximo: {np.max(error):.3f} m")
    print(f"Error final: {error[-1]:.3f} m")

print(
    "\nLandmarks reales:",
    len(mapa_slam)
)

print(
    "Landmarks estimados por SLAM:",
    len(ekf_slam.landmarks)
)
