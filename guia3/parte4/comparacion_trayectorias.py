#!/usr/bin/env python3
"""
Comparación directa entre trayectoria quíntica y trapezoidal.
"""

from roboticstoolbox import quintic, trapezoidal
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 200)

# Generar ambas trayectorias
q_traj = quintic(0, 1, t)
t_traj = trapezoidal(0, 1, t)

# --- Tabla comparativa ---
print("=" * 70)
print("COMPARACIÓN: QUÍNTICA vs TRAPEZOIDAL")
print("=" * 70)
print(f"{'Métrica':<25} {'Quíntica':>15} {'Trapezoidal':>15}")
print("-" * 70)
print(f"{'Duración (s)':<25} {1.0:>15.4f} {1.0:>15.4f}")
print(f"{'Posición final':<25} {q_traj.q[-1]:>15.4f} {t_traj.q[-1]:>15.4f}")
print(f"{'Velocidad pico':<25} {np.max(np.abs(q_traj.qd)):>15.4f} {np.max(np.abs(t_traj.qd)):>15.4f}")
print(f"{'Aceleración pico':<25} {np.max(np.abs(q_traj.qdd)):>15.4f} {np.max(np.abs(t_traj.qdd)):>15.4f}")

# Calcular jerk (derivada de la aceleración) numéricamente
jerk_q = np.gradient(q_traj.qdd, t)
jerk_t = np.gradient(t_traj.qdd, t)
print(f"{'Jerk pico':<25} {np.max(np.abs(jerk_q)):>15.4f} {np.max(np.abs(jerk_t)):>15.4f}")

# --- Gráficas comparativas ---
fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

# Posición
axes[0].plot(t, q_traj.q, 'b-', linewidth=2, label='Quíntica')
axes[0].plot(t, t_traj.q, 'r--', linewidth=2, label='Trapezoidal')
axes[0].set_ylabel('Posición q(t)')
axes[0].set_title('Comparación: Quíntica vs Trapezoidal')
axes[0].legend()
axes[0].grid(True)

# Velocidad
axes[1].plot(t, q_traj.qd, 'b-', linewidth=2, label='Quíntica')
axes[1].plot(t, t_traj.qd, 'r--', linewidth=2, label='Trapezoidal')
axes[1].set_ylabel('Velocidad q̇(t)')
axes[1].legend()
axes[1].grid(True)

# Aceleración
axes[2].plot(t, q_traj.qdd, 'b-', linewidth=2, label='Quíntica')
axes[2].plot(t, t_traj.qdd, 'r--', linewidth=2, label='Trapezoidal')
axes[2].set_ylabel('Aceleración q̈(t)')
axes[2].legend()
axes[2].grid(True)

# Jerk
axes[3].plot(t, jerk_q, 'b-', linewidth=2, label='Quíntica')
axes[3].plot(t, jerk_t, 'r--', linewidth=2, label='Trapezoidal')
axes[3].set_ylabel('Jerk (dq̈/dt)')
axes[3].set_xlabel('Tiempo (s)')
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.savefig('comparacion.png', dpi=100)
plt.show()

print("\nGráfica guardada como comparacion.png")
