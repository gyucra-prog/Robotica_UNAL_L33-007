#!/usr/bin/env python3
"""
Actividad 4: Trayectoria quíntica
Genera una trayectoria suave de 0 a 1 en 1 segundo.
"""

from roboticstoolbox import quintic
import numpy as np
import matplotlib.pyplot as plt

# Tiempo: 50 muestras entre 0 y 1 segundo
t = np.linspace(0, 1, 50)

# Trayectoria quíntica: posición inicial 0, final 1
traj = quintic(0, 1, t)

# --- Imprimir resultados ---
print("=" * 60)
print("TRAYECTORIA QUÍNTICA")
print("=" * 60)
print(f"{'t (s)':>8} {'q (pos)':>12} {'qd (vel)':>12} {'qdd (acel)':>14}")
print("-" * 60)
for i in range(0, len(t), 5):
    print(f"{t[i]:>8.3f} {traj.q[i]:>12.4f} {traj.qd[i]:>12.4f} {traj.qdd[i]:>14.4f}")

print("\n--- Valores en los extremos ---")
print(f"t=0.0  -> q={traj.q[0]:.4f},  qd={traj.qd[0]:.4f},  qdd={traj.qdd[0]:.4f}")
print(f"t=1.0  -> q={traj.q[-1]:.4f}, qd={traj.qd[-1]:.4f}, qdd={traj.qdd[-1]:.4f}")
print(f"\nVelocidad pico: {np.max(np.abs(traj.qd)):.4f} (unidades/s)")
print(f"Aceleración pico: {np.max(np.abs(traj.qdd)):.4f} (unidades/s²)")

# --- Gráfica ---
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(t, traj.q, 'b-', linewidth=2)
axes[0].set_ylabel('Posición q(t)')
axes[0].set_title('Trayectoria Quíntica: posición, velocidad y aceleración')
axes[0].grid(True)

axes[1].plot(t, traj.qd, 'r-', linewidth=2)
axes[1].set_ylabel('Velocidad q̇(t)')
axes[1].grid(True)

axes[2].plot(t, traj.qdd, 'g-', linewidth=2)
axes[2].set_ylabel('Aceleración q̈(t)')
axes[2].set_xlabel('Tiempo (s)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('quintica.png', dpi=100)
plt.show()

print("\nGráfica guardada como quintica.png")
