from scipy.spatial.transform import Rotation as R
import numpy as np

q = R.identity()
dt = 0.01
w = [0.1, 0.2, 0.3]

for i in range(200):
    dq = R.from_rotvec(np.array(w) * dt)
    q = q * dq

print(q.as_quat())  # Salida: [x, y, z, w]
