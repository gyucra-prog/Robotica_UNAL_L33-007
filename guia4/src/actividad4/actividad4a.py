import matplotlib.pyplot as plt
import numpy as np

# Trayectoria objetivo por waypoints
path = np.array([[0, 0], [2, 4], [6, 5], [8, 10]])

# Parámetros fijos del robot y simulación
dt = 0.1
L = 2.0
v = 0.5
iteraciones = 285


def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# Valores a variar
lookaheads = [0.2, 0.8, 2.0]  # Distancias de cambio de waypoint
Kds = [0.5, 1.3, 2.5]  # Ganancias del controlador angular

# Crear figura de 3x3 subgráficas
fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")

for i, lookahead in enumerate(lookaheads):
    for j, Kd in enumerate(Kds):
        # Estado inicial
        x, y, theta = 0.0, 0.0, 0.0
        current_index = 0

        X, Y = [], []

        # Simulación del seguimiento de puntos
        for _ in range(iteraciones):
            # Distancia al waypoint objetivo actual
            dx = path[current_index][0] - x
            dy = path[current_index][1] - y
            dist = np.hypot(dx, dy)

            # Avanzar al siguiente punto si estamos dentro del radio de lookahead
            if dist < lookahead and current_index < len(path) - 1:
                current_index += 1

            # Objetivo actual
            target = path[current_index]

            # Dirección hacia el objetivo
            dx = target[0] - x
            dy = target[1] - y
            theta_goal = np.arctan2(dy, dx)

            # Error angular y control
            error = angle_wrap(theta_goal - theta)
            delta = Kd * error

            # Saturación de dirección (máximo ±45° para evitar comportamientos no físicos)
            delta = np.clip(delta, -np.radians(45), np.radians(45))

            # Modelo cinemático (Euler)
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            theta += (v / L) * np.tan(delta) * dt

            X.append(x)
            Y.append(y)

        # Configurar la subgráfica correspondiente
        ax = axes[i, j]

        # Graficar trayectoria objetivo, puntos objetivo y ruta del robot
        ax.plot(
            path[:, 0],
            path[:, 1],
            "--",
            color="gray",
            linewidth=1.5,
            label="Ruta deseada",
        )
        ax.scatter(
            path[:, 0],
            path[:, 1],
            color="red",
            s=25,
            zorder=3,
            label="Waypoints",
        )
        ax.plot(X, Y, color="navy", linewidth=2, label="Robot")

        # Título formateado en 2 líneas compactas para evitar solapamiento
        ax.set_title(
            f"Lookahead = {lookahead} m\n$K_d$ = {Kd}",
            fontsize=9.5,
            fontweight="bold",
            pad=6,
        )

        ax.set_xlabel("x [m]", fontsize=8.5)
        ax.set_ylabel("y [m]", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=8)
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.6)

        # Leyenda visible únicamente en el primer cuadro
        if i == 0 and j == 0:
            ax.legend(fontsize=8, loc="upper left")

plt.show()
