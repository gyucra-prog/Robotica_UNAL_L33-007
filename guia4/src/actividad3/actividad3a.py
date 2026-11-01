import matplotlib.pyplot as plt
import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# Parámetros de la línea: ax + by + c = 0
a, b, c = 1.0, -2.0, 4.0

# Parámetros del robot y simulación
dt = 0.1
L = 2.0
v = 1.0
iteraciones = 250

# Variaciones a comparar
# 1. Tres poses iniciales (distintas distancias a la línea)
poses_iniciales = [
    (8.0, 5.0),  # Distancia moderada
    (15.0, 2.0),  # Mayor distancia
    (2.0, 10.0),  # Distancia en el otro lado de la línea
]

# 2. Tres combinaciones de ganancias (Kd, Kh)
ganancias = [
    (0.2, 0.5),  # Respuesta lenta / suave
    (0.5, 1.0),  # Configuración base
    (1.5, 2.0),  # Respuesta rápida / agresiva
]

# Generación de la línea objetivo para graficar
x_line = np.linspace(-2, 20, 100)
y_line = (-a * x_line - c) / b if b != 0 else np.linspace(-2, 20, 100)

# Crear figura de 3x3 subgráficas
fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")

for i, (x0, y0) in enumerate(poses_iniciales):
    # Calculamos la distancia inicial exacta para mostrarla en el título
    dist_inicial = (a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)

    for j, (Kd, Kh) in enumerate(ganancias):
        # Reiniciar estado inicial
        x, y = x0, y0
        theta = np.pi / 2  # Orientación inicial hacia arriba

        X, Y = [], []

        # Simulación del control de seguimiento de línea
        for _ in range(iteraciones):
            # Distancia con signo a la línea
            distance = (a * x + b * y + c) / np.sqrt(a**2 + b**2)

            # Orientación deseada de la línea
            theta_line = np.arctan2(a, -b)

            # Error angular
            error_theta = wrap_angle(theta_line - theta)

            # Acción de control (saturación opcional de delta para realismo)
            delta = Kd * distance + Kh * error_theta
            delta = np.clip(delta, -np.radians(45), np.radians(45))

            # Modelo cinemático
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            theta += (v / L) * np.tan(delta) * dt

            X.append(x)
            Y.append(y)

        # Configurar subgráfica
        ax = axes[i, j]

        # Trayectoria del robot y línea objetivo
        ax.plot(X, Y, color="navy", linewidth=2, label="Robot")
        ax.plot(
            x_line,
            y_line,
            "--",
            color="crimson",
            linewidth=1.5,
            label="Línea objetivo",
        )
        ax.plot(x0, y0, "go", markersize=5, label="Inicio")  # Punto de inicio

        # Título compacto en 2 líneas para evitar solapamientos
        ax.set_title(
            f"Pos ($x_0={x0:.0f}, y_0={y0:.0f}$) | Dist: {dist_inicial:.1f} m\n"
            f"$K_d$ = {Kd} | $K_h$ = {Kh}",
            fontsize=9.5,
            fontweight="bold",
            pad=6,
        )

        ax.set_xlabel("x [m]", fontsize=8.5)
        ax.set_ylabel("y [m]", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=8)
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.6)

        # Leyenda solo en la primera subgráfica para no saturar la vista
        if i == 0 and j == 0:
            ax.legend(fontsize=8, loc="upper left")

plt.show()
