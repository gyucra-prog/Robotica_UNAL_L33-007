import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np

class IMUIntegrator(Node):
    def __init__(self):
        super().__init__('imu_integrator')
        self.sub = self.create_subscription(Imu, '/imu', self.callback, 10)
        self.br = TransformBroadcaster(self)
        self.q = R.identity()
        self.last_time = self.get_clock().now()

    def callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time

        w = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        if np.linalg.norm(w) > 1e-6:  # Evitar rotaciones nulas
            dq = R.from_rotvec(w * dt)
            self.q = self.q * dq
            # Re-normalizar para prevenir drift numérico
            self.q = R.from_quat(self.q.as_quat() / np.linalg.norm(self.q.as_quat()))

        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'imu_link'

        q = self.q.as_quat()  # [x, y, z, w]
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.br.sendTransform(t)

def main():
    rclpy.init()
    node = IMUIntegrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
