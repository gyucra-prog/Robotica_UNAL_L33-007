import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================
# ACTIVIDAD 2
# No conmutatividad de las rotaciones
# ==========================================

# Rotación de 90° alrededor de Z
r1 = R.from_euler('z', 90, degrees=True)

# Rotación de 90° alrededor de X
r2 = R.from_euler('x', 90, degrees=True)

# Composición R1 * R2
r12 = r1 * r2

# Composición R2 * R1
r21 = r2 * r1

# Obtener matrices
R12 = r12.as_matrix()
R21 = r21.as_matrix()

print("=== ACTIVIDAD 2 ===")

print("\nR1 = Rotación de 90° alrededor de Z")
print(r1.as_matrix())

print("\nR2 = Rotación de 90° alrededor de X")
print(r2.as_matrix())

print("\nR1 * R2:")
print(R12)

print("\nR2 * R1:")
print(R21)

# Verificar si son iguales
print("\n¿R1*R2 = R2*R1?")
print(np.allclose(R12, R21))
