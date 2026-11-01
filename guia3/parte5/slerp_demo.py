#!/usr/bin/env python3
"""
Actividad 6: Interpolación de orientación con SLERP.
Interpola entre dos orientaciones (0° y 90° en Z).
"""

from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np

# --- Orientaciones de inicio y fin ---
r1 = R.from_euler('z', 0, degrees=True)
r2 = R.from_euler('z', 90, degrees=True)

print("=" * 60)
print("INTERPOLACIÓN SLERP: 0° -> 90° en Z")
print("=" * 60)
print(f"Rotación inicial: {r1.as_euler('xyz', degrees=True)}")
print(f"Rotación final:   {r2.as_euler('xyz', degrees=True)}")
print()

# --- Construir el interpolador SLERP ---
slerp = Slerp([0, 1], R.concatenate([r1, r2]))

# --- Muestrear 5 puntos entre 0 y 1 ---
times = np.linspace(0, 1, 5)
interp = slerp(times)

# --- Mostrar resultados en Euler ---
print(f"{'t':>6} {'Roll (°)':>10} {'Pitch (°)':>10} {'Yaw (°)':>10}")
print("-" * 40)
euler = interp.as_euler('xyz', degrees=True)
for t_val, (roll, pitch, yaw) in zip(times, euler):
    print(f"{t_val:>6.2f} {roll:>10.3f} {pitch:>10.3f} {yaw:>10.3f}")

# --- Mostrar cuaterniones ---
print("\n--- Cuaterniones interpolados [x, y, z, w] ---")
quats = interp.as_quat()
for t_val, q in zip(times, quats):
    print(f"t={t_val:.2f}: [{q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f}]")

# --- Verificar que el ángulo crece linealmente ---
print("\n--- Verificación de linealidad ---")
angles = np.array([np.linalg.norm(R.from_quat(q).as_rotvec()) * 180 / np.pi for q in quats])
print(f"Ángulos (grados): {angles}")
print(f"Diferencia entre pasos: {np.diff(angles)}")
print("(Si son aproximadamente iguales, la interpolación es uniforme)")
