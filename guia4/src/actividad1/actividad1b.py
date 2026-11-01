import matplotlib.pyplot as plt
import numpy as np

# Parámetros generales
v = 1.0  # Velocidad constante [m/s]
dt = 0.1  # Paso de tiempo [s]
pasos = 100  # Número de pasos de simulación

# 3 valores de delta (incluyendo delta = 0) y 3 valores de L
deltas = [0.3, 0.8, 0.0]  # Ángulos de dirección [rad]
Ls = [0, 4, 10]  # Longitudes del robot [m]

# Matriz de 3x3 (9 subgráficas en total)
fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")

for i, delta in enumerate(deltas):
    for j, L in enumerate(Ls):
        # Estado inicial
        x, y, theta = 0.0, 0.0, 0.0
        X, Y = [], []

        # Simulación
        for _ in range(pasos):
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta)

            # Manejo de L = 0 o delta = 0 (movimiento rectilíneo)
            if L == 0 or delta == 0:
                theta_dot = 0.0
            else:
                theta_dot = (v / L) * np.tan(delta)

            # Integración de Euler
            x += x_dot * dt
            y += y_dot * dt
            theta += theta_dot * dt

            X.append(x)
            Y.append(y)

        # Seleccionar subgráfica en la matriz
        ax = axes[i, j]

        # Estilo de línea según el caso
        color = "crimson" if delta == 0 else "navy"
        ax.plot(X, Y, color=color, linewidth=2)

        # Título formateado en 2 líneas compactas para evitar solapamiento
        ax.set_title(
            f"$\delta$ = {delta} rad\nL = {L} m",
            fontsize=10,
            fontweight="bold",
            pad=6,
        )

        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.6)

plt.show()
