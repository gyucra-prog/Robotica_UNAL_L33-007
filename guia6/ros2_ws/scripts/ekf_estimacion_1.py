import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import Bicycle, EKF, LandmarkMap, RangeBearingSensor
 
# =============================================================================
# 1. Trayectoria deseada (waypoints). Forma: [[x0, x1, x2], [y0, y1, y2]]
# =============================================================================
waypoints = np.array([
    [0.0, 5.0, 10.0],   # X
    [0.0, 8.0,  2.0],   # Y
    [0.0, 12.0,  3.0],
])
 
# =============================================================================
# 2. Parámetros de simulación
# =============================================================================
T = 25.0                                        # tiempo de simulación [s]
V = np.diag([0.02, np.deg2rad(0.5)]) ** 2       # covarianza de odometría
W = np.diag([0.10, np.deg2rad(1.0)]) ** 2       # covarianza del sensor (rango, rumbo)
P0 = np.diag([0.05, 0.05, np.deg2rad(0.5)]) ** 2  # incertidumbre inicial
x0 = np.array([waypoints[0, 0], waypoints[1, 0], 0.0])  # [x, y, theta]
WORKSPACE = [-2, 12]                            # área de trabajo (x e y)
 
# Mapa de landmarks (conocido) para que el EKF pueda corregirse
landmark_map = LandmarkMap(20, workspace=WORKSPACE, seed=0)
 
 
# =============================================================================
# 3. Controlador de persecución de punto
# =============================================================================
class WaypointController:
    """
    Persecución de punto sobre una lista de waypoints.
 
    - Usa la POSE ESTIMADA por el EKF (no la real), como en un robot real.
    - Se reinicia solo al comenzar cada simulación (t == 0), por lo que se
      puede volver a ejecutar sin reiniciar el kernel.
    - Firma compatible con Vehicle: f(vehicle, t, x).
    """
 
    def __init__(self, waypoints, v_max=1.0, capture_radius=0.8,
                 goal_tol=0.3, gamma_max=np.pi / 4):
        self.wp = waypoints
        self.v_max = v_max
        self.capture_radius = capture_radius   # radio para pasar al siguiente punto
        self.goal_tol = goal_tol               # tolerancia para detenerse al final
        self.gamma_max = gamma_max             # ángulo de dirección máximo
        self.ekf = None                        # se asigna después de crear el EKF
        self.idx = 1                           # el waypoint 0 es el punto de partida
 
    def __call__(self, vehicle, t, x_true):
        if t == 0:                             # inicio de una nueva simulación
            self.idx = 1
 
        # El controlador solo conoce la estimación del filtro
        pose = self.ekf.x_est[:3] if self.ekf is not None else x_true
        x, y, theta = pose
 
        last = self.wp.shape[1] - 1
        dx, dy = self.wp[0, self.idx] - x, self.wp[1, self.idx] - y
        dist = np.hypot(dx, dy)
 
        # Avanzar al siguiente waypoint al entrar en el radio de captura
        while self.idx < last and dist < self.capture_radius:
            self.idx += 1
            dx, dy = self.wp[0, self.idx] - x, self.wp[1, self.idx] - y
            dist = np.hypot(dx, dy)
 
        # Último punto alcanzado: detener el robot
        if self.idx == last and dist < self.goal_tol:
            return np.array([0.0, 0.0])
 
        # Error de rumbo normalizado a [-pi, pi]
        heading_error = np.arctan2(dy, dx) - theta
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
 
        # Velocidad constante; en el último tramo se reduce al acercarse
        v = self.v_max
        if self.idx == last:
            v = np.clip(dist, 0.3, self.v_max)
 
        gamma = np.clip(heading_error, -self.gamma_max, self.gamma_max)
        return np.array([v, gamma])
 
 
# =============================================================================
# 4. Simulación: con sensor (localización con mapa) o solo odometría
# =============================================================================
def simulate(use_sensor):
    robot = Bicycle(covar=V, x0=x0, workspace=WORKSPACE, animation=None)
    controller = WaypointController(waypoints)
    robot.control = controller
 
    if use_sensor:
        sensor = RangeBearingSensor(robot, landmark_map, covar=W,
                                    range=(0.5, 6.0), angle=np.pi / 2)
        ekf = EKF(robot=(robot, V), P0=P0, sensor=(sensor, W),
                  map=landmark_map, animate=False)
    else:
        ekf = EKF(robot=(robot, V), P0=P0, animate=False)  # solo predicción
 
    controller.ekf = ekf
    ekf.run(T=T)
    return robot, ekf
 
 
robot, ekf = simulate(use_sensor=True)         # EKF con corrección por sensor
robot_dr, ekf_dr = simulate(use_sensor=False)  # referencia: solo odometría
 
 
# =============================================================================
# 5. Extracción de datos
# =============================================================================
def extract(robot, ekf):
    real = np.asarray(robot.x_hist)             # (N, 3) pose real
    est = ekf.get_xyt()                         # (N, 3) pose estimada
    t = ekf.get_t()                             # (N,)   tiempo
    pnorm = ekf.get_Pnorm()                     # (N,)   sqrt(det(P))
    err = np.hypot(est[:, 0] - real[:, 0], est[:, 1] - real[:, 1])
    return real, est, t, pnorm, err
 
 
real, est, t, pnorm, err = extract(robot, ekf)
real_dr, est_dr, t_dr, pnorm_dr, err_dr = extract(robot_dr, ekf_dr)
 
# Resumen numérico
for name, r, e in (("EKF + sensor", real, err), ("Solo odometría", real_dr, err_dr)):
    d_mid = np.min(np.hypot(r[:, 0] - waypoints[0, 1], r[:, 1] - waypoints[1, 1]))
    d_end = np.hypot(r[-1, 0] - waypoints[0, -1], r[-1, 1] - waypoints[1, -1])
    print(f"{name:15s} | dist. mín. al punto intermedio: {d_mid:5.2f} m | "
          f"dist. final al objetivo: {d_end:5.2f} m | "
          f"error de estimación medio: {e.mean():5.3f} m, final: {e[-1]:5.3f} m")
 
# =============================================================================
# 6. Gráfica 1: trayectorias (ideal vs real vs estimada por EKF)
# =============================================================================
plt.figure(figsize=(9, 7))
plt.plot(waypoints[0], waypoints[1], 'k--', linewidth=1.5, label='Trayectoria ideal (waypoints)')
plt.plot(real[:, 0], real[:, 1], 'b-', linewidth=2, label='Trayectoria real')
plt.plot(est[:, 0], est[:, 1], 'r--', linewidth=2, label='Estimada (EKF + sensor)')
plt.plot(est_dr[:, 0], est_dr[:, 1], color='orange', linestyle=':', linewidth=2,
         label='Estimada (solo odometría)')
plt.plot(real_dr[:, 0], real_dr[:, 1], color='gray', linestyle='-', linewidth=1,
         label='Real (solo odometría)')
 
lm = landmark_map.landmarks                     # (2, N)
plt.plot(lm[0], lm[1], 'g*', markersize=8, label='Landmarks')
 
plt.plot(waypoints[0, 0], waypoints[1, 0], 'go', markersize=10, label='Inicio')
plt.plot(waypoints[0, 1], waypoints[1, 1], 'ko', markersize=6, label='Punto intermedio')
plt.plot(waypoints[0, -1], waypoints[1, -1], 'rx', markersize=10,
         markeredgewidth=3, label='Fin')
 
plt.title('Comparación de trayectoria: ideal vs real vs EKF')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.legend(loc='best', fontsize=8)
plt.grid(True)
plt.axis('equal')
 
# =============================================================================
# 7. Gráfica 2: incertidumbre y error de estimación
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
 
ax1.semilogy(t, pnorm, linewidth=2, color='purple', label='EKF + sensor')
ax1.semilogy(t_dr, pnorm_dr, linewidth=2, color='orange', linestyle='--',
             label='Solo odometría')
ax1.set_ylabel(r'$\sqrt{\det P}$ (incertidumbre)')
ax1.set_title('Evolución de la incertidumbre en el EKF')
ax1.legend()
ax1.grid(True, which='both')
 
ax2.plot(t, err, linewidth=2, color='purple', label='EKF + sensor')
ax2.plot(t_dr, err_dr, linewidth=2, color='orange', linestyle='--', label='Solo odometría')
ax2.set_xlabel('Tiempo [s]')
ax2.set_ylabel('Error de posición [m]')
ax2.set_title('Error entre pose estimada y pose real')
ax2.legend()
ax2.grid(True)
 
plt.tight_layout()
plt.show()
