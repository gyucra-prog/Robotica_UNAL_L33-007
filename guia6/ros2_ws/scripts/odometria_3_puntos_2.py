from roboticstoolbox import Bicycle
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración del robot y ruido
dt = 0.1  # Paso de tiempo de integración (s)
V = np.diag([0.05, np.deg2rad(0.5)])**2  # Varianza ajustada a escala temporal

# Inicializar modelos de robot con dt explícito
robot_ruidoso = Bicycle(covar=V, dt=dt)
robot_ideal = Bicycle(covar=np.zeros((2, 2)), dt=dt)

# 2. Definición de puntos y estado inicial
puntos = np.array([
    [4.0, 0.0],
    [4.0, 3.0],
    [0.0, 3.0]
])

pose_real = np.array([0.0, 0.0, 0.0])
pose_ideal = np.array([0.0, 0.0, 0.0])

trayectoria_real = [pose_real.copy()]
trayectoria_ideal = [pose_ideal.copy()]

# 3. Parámetros del controlador
velocidad = 0.5            # m/s
distancia_minima = 0.25     # Tolerancia aumentada para garantizar captura
gamma_max = np.deg2rad(30) # Ángulo máximo de dirección
K_p = 2.0                  # Ganancia proporcional de dirección
max_iter_por_punto = 500   # Evita bucles infinitos por seguridad

# 4. Bucle de control
for i, punto in enumerate(puntos):
    iteraciones = 0
    
    while iteraciones < max_iter_por_punto:
        iteraciones += 1

        # Distancia al punto objetivo desde la pose ideal
        dx = punto[0] - pose_ideal[0]
        dy = punto[1] - pose_ideal[1]
        distancia = np.hypot(dx, dy)

        # Criterio de parada para el waypoint actual
        if distancia < distancia_minima:
            print(f"Waypoint {i+1} alcanzado.")
            break

        # Error angular respecto al objetivo
        angulo_objetivo = np.arctan2(dy, dx)
        error_angular = angulo_objetivo - pose_ideal[2]
        error_angular = np.arctan2(np.sin(error_angular), np.cos(error_angular))

        # Control proporcional de dirección (Gamma)
        gamma = K_p * error_angular
        gamma = np.clip(gamma, -gamma_max, gamma_max)

        # Actualización del robot ideal
        odo_ideal = robot_ideal.step((velocidad, gamma))
        pose_ideal = robot_ideal.f(pose_ideal, odo_ideal)

        # Actualización del robot ruidoso
        odo_real = robot_ruidoso.step((velocidad, gamma))
        pose_real = robot_ruidoso.f(pose_real, odo_real)

        # Guardar historial
        trayectoria_ideal.append(pose_ideal.copy())
        trayectoria_real.append(pose_real.copy())

# Convertir historiales a arrays NumPy
trayectoria_ideal = np.array(trayectoria_ideal)
trayectoria_real = np.array(trayectoria_real)

# 5. Cálculo del error acumulado
error_x = trayectoria_real[:, 0] - trayectoria_ideal[:, 0]
error_y = trayectoria_real[:, 1] - trayectoria_ideal[:, 1]
error_posicion = np.sqrt(error_x**2 + error_y**2)

# 6. Gráficas y Resultados
plt.figure(figsize=(10, 6))
plt.plot(trayectoria_ideal[:, 0], trayectoria_ideal[:, 1], label="Robot ideal", linewidth=2)
plt.plot(trayectoria_real[:, 0], trayectoria_real[:, 1], label="Robot con ruido", linestyle="--", linewidth=2)
plt.scatter(puntos[:, 0], puntos[:, 1], color="red", marker="x", s=100, label="Waypoints")

for i, punto in enumerate(puntos):
    plt.annotate(f"P{i+1}", (punto[0], punto[1]), xytext=(8, 8), textcoords="offset points")

plt.scatter(0, 0, color="black", marker="o", label="Inicio")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Odometría: Trayectoria con modelo Bicycle")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()

# Gráfica del error
plt.figure(figsize=(10, 4))
plt.plot(error_posicion, color="orange", label="Error de posición")
plt.xlabel("Pasos de simulación")
plt.ylabel("Error [m]")
plt.title("Divergencia por acumulativo de ruido de odometría")
plt.grid(True)
plt.legend()
plt.show()

# Imprimir métricas finales
print("\n--- Resultados de la Simulación ---")
print("Pose final ideal [x, y, theta]:", np.round(trayectoria_ideal[-1], 3))
print("Pose final real  [x, y, theta]:", np.round(trayectoria_real[-1], 3))
print("Error final de posición [m]:", round(error_posicion[-1], 3))
print("Error máximo registrado [m]:", round(np.max(error_posicion), 3))
