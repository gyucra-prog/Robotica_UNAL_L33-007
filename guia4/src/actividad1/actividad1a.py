import matplotlib.pyplot as plt
import numpy as np

# Parámetros generales
v = 1.0
dt = 0.1
pasos = 100

# 6 combinaciones
casos = [(0.3, 0), (0.3, 4), (0.3, 10), (0.8, 0), (0.8, 4), (0.8, 10)]

# Opción 1: layout="constrained" ajusta automáticamente márgenes y títulos
fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")

for idx, (delta, L) in enumerate(casos):
    row, col = divmod(idx, 3)
    ax = axes[row, col]

    x, y, theta = 0.0, 0.0, 0.0
    X, Y = [], []

    for _ in range(pasos):
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = 0.0 if L == 0 else (v / L) * np.tan(delta)

        x += x_dot * dt
        y += y_dot * dt
        theta += theta_dot * dt

        X.append(x)
        Y.append(y)

    ax.plot(X, Y, color="navy", linewidth=2)

    # Título en dos líneas con pad adicional para separar del marco
    ax.set_title(
        f"$\delta$ = {delta} rad\nL = {L} m", fontsize=10, fontweight="bold", pad=8
    )

    ax.set_xlabel("x [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.6)

plt.show()
