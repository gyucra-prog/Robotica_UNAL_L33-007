import matplotlib.pyplot as plt
import numpy as np
from roboticstoolbox import Bicycle

# =========================================
# 1. Configuración del ruido
# =========================================
V = np.diag([0.2, np.deg2rad(0.5)]) ** 2

robot_ruidoso = Bicycle(covar=V)
robot_ideal = Bicycle(covar=np.zeros((2, 2)))

# =========================================
# 2. Definición de puntos y estado
# =========================================
puntos = np.array([[4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])

pose_real = np.array([0.0, 0.0, 0.0])
pose_ideal = np.array([0.0, 0.0, 0.0])

trayectoria_real = [pose_real.copy()]
trayectoria_ideal = [pose_ideal.copy()]

# =========================================
# 3. Parámetros del controlador
# =========================================
velocidad = 0.5
distancia_minima = 0.15
L = 1.0  # Distancia entre ejes
gamma_max = np.deg2rad(30)
max_iter_por_punto = 1000

# =========================================
# 4. Bucle de control
# =========================================
for i, punto in enumerate(puntos):
    iteraciones = 0

    while iteraciones < max_iter_por_punto:
        iteraciones += 1

        # Error de posición respecto a la pose ideal
        dx = punto[0] - pose_ideal[0]
        dy = punto[1] - pose_ideal[1]
        distancia = np.sqrt(dx**2 + dy**2)

        if distancia < distancia_minima:
            print(
                f"Punto P{i+1} alcanzado en {iteraciones} pasos de integración."
            )
            break

        # Control angular
        angulo_objetivo = np.arctan2(dy, dx)
        error_angular = angulo_objetivo - pose_ideal[2]
        error_angular = np.arctan2(
            np.sin(error_angular), np.cos(error_angular)
        )

        # Ángulo de dirección
        gamma = np.arctan2(
            2 * L * np.sin(error_angular), max(distancia, 0.1)
        )
        gamma = np.clip(gamma, -gamma_max, gamma_max)

        # Actualización de estados
        odo_ideal = robot_ideal.step((velocidad, gamma))
        pose_ideal = robot_ideal.f(pose_ideal, odo_ideal)

        odo_real = robot_ruidoso.step((velocidad, gamma))
        pose_real = robot_ruidoso.f(pose_real, odo_real)

        trayectoria_ideal.append(pose_ideal.copy())
        trayectoria_real.append(pose_real.copy())

# =========================================
# 5. Cálculo de métricas
# =========================================
trayectoria_ideal = np.array(trayectoria_ideal)
trayectoria_real = np.array(trayectoria_real)

error_posicion = np.linalg.norm(
    trayectoria_real[:, :2] - trayectoria_ideal[:, :2], axis=1
)

# =========================================
# 6. Gráficas
# =========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Trayectoria
ax1.plot(
    trayectoria_ideal[:, 0],
    trayectoria_ideal[:, 1],
    "b-",
    label="Robot Ideal",
    linewidth=2,
)
ax1.plot(
    trayectoria_real[:, 0],
    trayectoria_real[:, 1],
    "r--",
    label="Robot con Ruido",
    linewidth=2,
)
ax1.scatter(
    puntos[:, 0], puntos[:, 1], c="g", marker="x", s=100, label="Waypoints"
)

for idx, p in enumerate(puntos):
    ax1.annotate(
        f"P{idx+1}", (p[0], p[1]), xytext=(5, 5), textcoords="offset points"
    )

ax1.set_xlabel("X [m]")
ax1.set_ylabel("Y [m]")
ax1.set_title("Comparativa de Trayectoria")
ax1.grid(True)
ax1.axis("equal")
ax1.legend()

# Error acumulado
ax2.plot(error_posicion, "k-", linewidth=1.5)
ax2.set_xlabel("Paso de simulación")
ax2.set_ylabel("Error Euclidiano [m]")
ax2.set_title("Evolución del Error Odométrico")
ax2.grid(True)

plt.tight_layout()
plt.show()

# =========================================
# 7. Imprimir resultados
# =========================================
print(f"\n--- Resumen ---")
print(f"Error Final:   {error_posicion[-1]:.3f} m")
print(f"Error Máximo:  {np.max(error_posicion):.3f} m")
