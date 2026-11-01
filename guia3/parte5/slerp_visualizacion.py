#!/usr/bin/env python3
"""
Visualización de la interpolación SLERP en 3D.
Compara SLERP vs interpolación lineal de ángulos de Euler.
"""

from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Dos orientaciones: una rotación compuesta ---
r1 = R.from_euler('xyz', [0, 0, 0], degrees=True)
r2 = R.from_euler('xyz', [45, 60, 120], degrees=True)

print("=" * 60)
print("SLERP: rotación compuesta")
print("=" * 60)
print(f"r1 (euler xyz): {r1.as_euler('xyz', degrees=True)}")
print(f"r2 (euler xyz): {r2.as_euler('xyz', degrees=True)}")

# --- Interpolación SLERP con 100 puntos ---
slerp = Slerp([0, 1], R.concatenate([r1, r2]))
times = np.linspace(0, 1, 100)
interp = slerp(times)
euler_slerp = interp.as_euler('xyz', degrees=True)

# --- Interpolación lineal de Euler (mal método, para comparar) ---
e1 = r1.as_euler('xyz', degrees=True)
e2 = r2.as_euler('xyz', degrees=True)
euler_linear = np.outer(1 - times, e1) + np.outer(times, e2)

# --- Gráfica 1: SLERP vs Euler lineal ---
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

for i, axis_name in enumerate(['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']):
    axes[0].plot(times, euler_slerp[:, i], linewidth=2, label=f'SLERP {axis_name}')
    axes[1].plot(times, euler_linear[:, i], '--', linewidth=2, label=f'Euler lineal {axis_name}')

axes[0].set_ylabel('Ángulo (°)')
axes[0].set_title('SLERP vs Interpolación lineal de Euler')
axes[0].legend(); axes[0].grid(True)

axes[1].set_ylabel('Ángulo (°)')
axes[1].set_xlabel('Fracción de interpolación t')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig('slerp_vs_euler.png', dpi=100)
plt.close()

# --- Gráfica 2: Trayectoria 3D de un vector unitario ---
# Aplicar cada rotación al vector (1,0,0) para ver el camino
vec = np.array([1, 0, 0])
points_slerp = interp.apply(vec)
points_euler = R.from_euler('xyz', euler_linear, degrees=True).apply(vec)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(points_slerp[:, 0], points_slerp[:, 1], points_slerp[:, 2],
         'b-', linewidth=2, label='SLERP')
ax1.scatter([1, 0], [0, 0], [0, 0], c='red', s=100, label='Inicio/Fin')
ax1.set_title('Trayectoria del vector (1,0,0) - SLERP')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(points_euler[:, 0], points_euler[:, 1], points_euler[:, 2],
         'r-', linewidth=2, label='Euler lineal')
ax2.scatter([1, 0], [0, 0], [0, 0], c='red', s=100, label='Inicio/Fin')
ax2.set_title('Trayectoria del vector (1,0,0) - Euler lineal')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.legend()

plt.tight_layout()
plt.savefig('trayectoria_3d.png', dpi=100)
plt.close()

# --- Verificar la norma de los cuaterniones ---
print("\n--- Norma de los cuaterniones SLERP ---")
norms = np.linalg.norm(interp.as_quat(), axis=1)
print(f"Norma mínima: {norms.min():.6f}")
print(f"Norma máxima: {norms.max():.6f}")
print(f"Desviación: {norms.max() - norms.min():.2e}")
print("(Debe ser ~1.0, confirmando que SLERP mantiene cuaterniones unitarios)")

# --- Velocidad angular ---
rotvecs = interp.as_rotvec()
angles = np.linalg.norm(rotvecs, axis=1) * 180 / np.pi
angular_vel = np.diff(angles) / np.diff(times)

print("\n--- Velocidad angular (grados por unidad de t) ---")
print(f"Velocidad media: {np.mean(angular_vel):.4f}")
print(f"Desviación estándar: {np.std(angular_vel):.6f}")
print("(Desviación ~0 significa velocidad angular constante)")

print("\nGráficas guardadas: slerp_vs_euler.png, trayectoria_3d.png")
