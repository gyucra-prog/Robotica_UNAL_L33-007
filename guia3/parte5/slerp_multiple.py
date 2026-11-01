#!/usr/bin/env python3
"""
SLERP con múltiples poses (waypoints).
Útil para trayectorias de orientación complejas.
"""

from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 4 waypoints de orientación ---
waypoints_deg = [
    [0,   0,   0],    # t=0
    [30,  0,   45],   # t=1
    [30,  60,  90],   # t=2
    [0,   0,   180],  # t=3
]
times_wp = [0, 1, 2, 3]

rotations = R.from_euler('xyz', waypoints_deg, degrees=True)

# --- SLERP con múltiples waypoints ---
slerp = Slerp(times_wp, rotations)
times = np.linspace(0, 3, 200)
interp = slerp(times)
euler = interp.as_euler('xyz', degrees=True)

# --- Gráfica ---
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

axes[0].plot(times, euler[:, 0], 'r-', linewidth=2, label='Roll')
axes[0].plot(times, euler[:, 1], 'g-', linewidth=2, label='Pitch')
axes[0].plot(times, euler[:, 2], 'b-', linewidth=2, label='Yaw')
for i, tw in enumerate(times_wp):
    axes[0].axvline(tw, color='gray', linestyle=':', alpha=0.5)
    axes[0].scatter([tw]*3, waypoints_deg[i], c=['red','green','blue'], s=60, zorder=5)
axes[0].set_ylabel('Ángulo (°)')
axes[0].set_title('SLERP con múltiples waypoints')
axes[0].legend(); axes[0].grid(True)

# Velocidad angular
rotvecs = interp.as_rotvec()
angles = np.linalg.norm(rotvecs, axis=1) * 180 / np.pi
angular_vel = np.gradient(angles, times)
axes[1].plot(times, angular_vel, 'k-', linewidth=2)
axes[1].set_ylabel('Velocidad angular (°/s)')
axes[1].set_xlabel('Tiempo (s)')
axes[1].set_title('Velocidad angular (constante entre waypoints)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('slerp_multiple.png', dpi=100)
plt.close()

# --- Tabla resumen ---
print("=" * 70)
print("SLERP MULTI-WAYPOINT")
print("=" * 70)
print(f"{'t':>6} {'Roll (°)':>10} {'Pitch (°)':>10} {'Yaw (°)':>10} {'ω (°/s)':>10}")
print("-" * 50)
for i in range(0, len(times), 20):
    print(f"{times[i]:>6.2f} {euler[i,0]:>10.2f} {euler[i,1]:>10.2f} {euler[i,2]:>10.2f} {angular_vel[i]:>10.2f}")

print("\nGráfica guardada: slerp_multiple.png")
