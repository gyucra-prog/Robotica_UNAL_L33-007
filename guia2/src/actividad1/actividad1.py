import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================
# ACTIVIDAD 1
# Rotación de 90° alrededor del eje Z
# ==========================================

# Crear la rotación
r = R.from_euler('z', 90, degrees=True)

# Obtener cuaternión
q = r.as_quat()  # [x, y, z, w]

# Obtener matriz de rotación
R_mat = r.as_matrix()

print("=== ACTIVIDAD 1 ===")

print("\nCuaternión [x, y, z, w]:")
print(q)

print("\nMatriz de rotación:")
print(R_mat)

# ==========================================
# Verificación de equivalencia
# ==========================================

# Crear una nueva rotación usando el cuaternión
r_from_q = R.from_quat(q)

# Obtener nuevamente la matriz
R_from_q = r_from_q.as_matrix()

print("\nMatriz reconstruida desde el cuaternión:")
print(R_from_q)

# Comparar las matrices
equivalentes = np.allclose(R_mat, R_from_q)

print("\n¿Las representaciones son equivalentes?")
print(equivalentes)
