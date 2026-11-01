from scipy.spatial.transform import Rotation as R
import numpy as np

# Intervalo de tiempo
dt = 0.01

# Velocidad angular [wx, wy, wz] en rad/s
w = np.array([0.0, 0.0, 1.0])

# Orientacion inicial
r = R.identity()

# Integracion de la orientacion
for i in range(100):
    r = r * R.from_rotvec(w * dt)

# Convertir la orientacion final a angulos de Euler
euler = r.as_euler('xyz', degrees=True)

print("Orientacion final:")
print("Roll  (X): {:.4f} grados".format(euler[0]))
print("Pitch (Y): {:.4f} grados".format(euler[1]))
print("Yaw   (Z): {:.4f} grados".format(euler[2]))
