import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from roboticstoolbox import (
    Bicycle,
    EKF,
    RandomPath,
    LandmarkMap,
    RangeBearingSensor
)

# ==========================================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================================

SEMILLA = 10
TIEMPO = 40

# Ruido del movimiento:
# Desviación estándar de velocidad y dirección.
V = np.diag([
    0.05,                  # Ruido de velocidad
    np.deg2rad(1.5)        # Ruido de dirección
])**2

# Ruido del sensor:
# Desviación estándar de distancia y ángulo.
W = np.diag([
    0.20,                  # Ruido de distancia (m)
    np.deg2rad(2.5)        # Ruido angular
])**2

# Incertidumbre inicial de la pose.
P0 = np.diag([
    0.10,
    0.10,
    np.deg2rad(2)
])**2


# ==========================================================
# 2. FUNCIÓN PARA CREAR Y EJECUTAR CADA MODALIDAD
# ==========================================================

def ejecutar_ekf(tipo):

    # Reiniciar semilla para facilitar la comparación.
    np.random.seed(SEMILLA)

    # Crear robot.
    robot = Bicycle(
        covar=V,
        animation=None,
        workspace=10
    )

    # Crear trayectoria.
    robot.control = RandomPath(
        workspace=10,
        seed=SEMILLA
    )

    # Crear mapa real de landmarks.
    mapa = LandmarkMap(
        20,
        workspace=10,
        seed=SEMILLA
    )

    # ======================================================
    # MODALIDAD 1: EKF SIN CORRECCIÓN
    # ======================================================

    if tipo == "odometria":

        ekf = EKF(
            robot=(robot, V),
            P0=P0
        )

    # ======================================================
    # MODALIDAD 2: EKF CON MAPA CONOCIDO
    # ======================================================

    elif tipo == "mapa_conocido":

        sensor = RangeBearingSensor(
            robot=robot,
            map=mapa,
            covar=W,
            angle=[-np.pi/2, np.pi/2],
            range=4
        )

        ekf = EKF(
            robot=(robot, V),
            P0=P0,
            map=mapa,
            sensor=(sensor, W)
        )

    # ======================================================
    # MODALIDAD 3: EKF-SLAM
    # ======================================================

    elif tipo == "slam":

        sensor = RangeBearingSensor(
            robot=robot,
            map=mapa,
            covar=W,
            angle=[-np.pi/2, np.pi/2],
            range=4
        )

        ekf = EKF(
            robot=(robot, V),
            P0=P0,
            sensor=(sensor, W),
            map=None
        )

    # Ejecutar simulación sin animación interna.
    ekf.run(
        T=TIEMPO,
        animate=False
    )

    return robot, mapa, ekf


# ==========================================================
# 3. EJECUTAR LAS TRES MODALIDADES
# ==========================================================

print("Ejecutando EKF sin corrección...")
robot_odom, mapa_odom, ekf_odom = ejecutar_ekf("odometria")

print("Ejecutando EKF con mapa conocido...")
robot_mapa, mapa_real, ekf_mapa = ejecutar_ekf("mapa_conocido")

print("Ejecutando EKF-SLAM...")
robot_slam, mapa_slam, ekf_slam = ejecutar_ekf("slam")

print("Simulaciones terminadas.")


# ==========================================================
# 4. OBTENER TRAYECTORIAS Y ERRORES
# ==========================================================

def obtener_datos(robot, ekf):

    # Trayectoria real del robot.
    real = robot.x_hist

    # Trayectoria estimada por el EKF.
    estimada = ekf.get_xyt()

    # Igualar cantidad de muestras.
    n = min(len(real), len(estimada))

    real = real[:n]
    estimada = estimada[:n]

    # Error de posición (distancia euclidiana).
    error = np.linalg.norm(
        real[:, :2] - estimada[:, :2],
        axis=1
    )

    # Incertidumbre de la pose.
    # Se utiliza únicamente el bloque de covarianza
    # correspondiente a x, y y orientación.
    incertidumbre = np.array([
        np.sqrt(max(0, np.trace(h.P[:3, :3])))
        for h in ekf.history[:n]
    ])

    tiempo = ekf.get_t()[:n]

    return real, estimada, error, incertidumbre, tiempo


datos_odom = obtener_datos(robot_odom, ekf_odom)
datos_mapa = obtener_datos(robot_mapa, ekf_mapa)
datos_slam = obtener_datos(robot_slam, ekf_slam)


# ==========================================================
# 5. CONFIGURAR FIGURA DE ANIMACIÓN
# ==========================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

fig.suptitle(
    "Comparación de EKF y EKF-SLAM con ruido",
    fontsize=16
)

configuraciones = [
    (ax1, datos_odom, "EKF simple"),
    (ax2, datos_mapa, "EKF con mapa"),
    (ax3, datos_slam, "EKF-SLAM")
]

lineas = []
marcadores = []

for ax, datos, titulo in configuraciones:

    real, estimada, error, incertidumbre, tiempo = datos

    ax.set_title(titulo)
    ax.set_xlabel("Posición X (m)")
    ax.set_ylabel("Posición Y (m)")

    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_aspect("equal")
    ax.grid(True)

    # Trayectoria real.
    linea_real, = ax.plot(
        [],
        [],
        "--",
        color="black",
        linewidth=1.5,
        label="Real"
    )

    # Trayectoria estimada.
    linea_estimada, = ax.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2,
        label="Estimada"
    )

    # Posición actual del robot.
    marcador, = ax.plot(
        [],
        [],
        "o",
        color="tab:red",
        markersize=7
    )

    ax.legend(loc="upper right")

    lineas.append((linea_real, linea_estimada))
    marcadores.append(marcador)


# ==========================================================
# 6. PANEL DE ERROR E INCERTIDUMBRE
# ==========================================================

ax4.set_title("Error de posición e incertidumbre")
ax4.set_xlabel("Tiempo (s)")
ax4.set_ylabel("Error de posición (m)")
ax4.grid(True)

# Errores.
linea_error_odom, = ax4.plot(
    [], [], label="Error - Odometría"
)

linea_error_mapa, = ax4.plot(
    [], [], label="Error - Mapa conocido"
)

linea_error_slam, = ax4.plot(
    [], [], label="Error - SLAM"
)

# Incertidumbre en un segundo eje.
ax4b = ax4.twinx()
ax4b.set_ylabel("Incertidumbre de pose")

linea_inc_odom, = ax4b.plot(
    [], [], "--", label="Incertidumbre - Odometría"
)

linea_inc_mapa, = ax4b.plot(
    [], [], "--", label="Incertidumbre - Mapa"
)

linea_inc_slam, = ax4b.plot(
    [], [], "--", label="Incertidumbre - SLAM"
)

# Unificar leyendas.
lineas_metricas = [
    linea_error_odom,
    linea_error_mapa,
    linea_error_slam,
    linea_inc_odom,
    linea_inc_mapa,
    linea_inc_slam
]

ax4.legend(
    lineas_metricas,
    [l.get_label() for l in lineas_metricas],
    loc="upper left",
    fontsize=8
)


# ==========================================================
# 7. PREPARAR DATOS PARA LA ANIMACIÓN
# ==========================================================

todos_los_datos = [
    datos_odom,
    datos_mapa,
    datos_slam
]

N = min(len(d[0]) for d in todos_los_datos)

tiempo = datos_odom[4][:N]

# Límites del gráfico de métricas.
ax4.set_xlim(tiempo[0], tiempo[-1])

max_error = max(
    np.max(d[2][:N]) for d in todos_los_datos
)

ax4.set_ylim(0, max(0.5, max_error * 1.2))

max_inc = max(
    np.max(d[3][:N]) for d in todos_los_datos
)

ax4b.set_ylim(0, max(0.01, max_inc * 1.2))


# ==========================================================
# 8. FUNCIÓN DE ACTUALIZACIÓN DE LA ANIMACIÓN
# ==========================================================

def actualizar(frame):

    # Actualizar las tres trayectorias.
    for i, (ax, datos, titulo) in enumerate(configuraciones):

        real, estimada, error, incertidumbre, t = datos

        real = real[:N]
        estimada = estimada[:N]

        # Trayectoria real hasta el instante actual.
        lineas[i][0].set_data(
            real[:frame + 1, 0],
            real[:frame + 1, 1]
        )

        # Trayectoria estimada hasta el instante actual.
        lineas[i][1].set_data(
            estimada[:frame + 1, 0],
            estimada[:frame + 1, 1]
        )

        # Posición actual estimada.
        marcadores[i].set_data(
            [estimada[frame, 0]],
            [estimada[frame, 1]]
        )

    # Actualizar errores.
    linea_error_odom.set_data(
        tiempo[:frame + 1],
        datos_odom[2][:frame + 1]
    )

    linea_error_mapa.set_data(
        tiempo[:frame + 1],
        datos_mapa[2][:frame + 1]
    )

    linea_error_slam.set_data(
        tiempo[:frame + 1],
        datos_slam[2][:frame + 1]
    )

    # Actualizar incertidumbres.
    linea_inc_odom.set_data(
        tiempo[:frame + 1],
        datos_odom[3][:frame + 1]
    )

    linea_inc_mapa.set_data(
        tiempo[:frame + 1],
        datos_mapa[3][:frame + 1]
    )

    linea_inc_slam.set_data(
        tiempo[:frame + 1],
        datos_slam[3][:frame + 1]
    )

    fig.suptitle(
        f"Comparación EKF y SLAM | Tiempo: {tiempo[frame]:.2f} s",
        fontsize=16
    )

    return []


# ==========================================================
# 9. INICIAR ANIMACIÓN
# ==========================================================

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
# 10. RESULTADOS NUMÉRICOS
# ==========================================================

print("\n========== RESULTADOS FINALES ==========")

nombres = [
    "EKF sin corrección",
    "EKF con mapa conocido",
    "EKF-SLAM"
]

for nombre, datos in zip(nombres, todos_los_datos):

    error = datos[2]
    incertidumbre = datos[3]

    print(f"\n{nombre}")
    print(f"Error medio: {np.mean(error):.4f} m")
    print(f"Error máximo: {np.max(error):.4f} m")
    print(f"Error final: {error[-1]:.4f} m")
    print(f"Incertidumbre final: {incertidumbre[-1]:.6f}")

print("\nLandmarks estimados por SLAM:",
      len(ekf_slam.landmarks))
