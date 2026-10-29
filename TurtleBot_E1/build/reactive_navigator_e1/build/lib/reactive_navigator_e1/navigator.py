import math
from statistics import median

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped


class ReactiveNavigator(Node):

    def __init__(self):
        super().__init__('reactive_navigator_e1')

        self.scan_data = None

        # Parámetros de seguridad y movimiento
        self.declare_parameter('safe_distance', 0.65)
        self.declare_parameter('clear_distance', 0.85)
        self.declare_parameter('linear_speed', 0.08)
        self.declare_parameter('angular_speed', 0.55)

        self.safe_distance = float(
            self.get_parameter('safe_distance').value
        )
        self.clear_distance = float(
            self.get_parameter('clear_distance').value
        )
        self.linear_speed = float(
            self.get_parameter('linear_speed').value
        )
        self.angular_speed = float(
            self.get_parameter('angular_speed').value
        )

        # Estado del controlador: mantiene el giro elegido
        self.avoiding = False
        self.turn_direction = 1.0
        self.clear_count = 0

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmd_publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            'Navegador reactivo iniciado.'
        )

    def scan_callback(self, msg):
        self.scan_data = msg

    @staticmethod
    def sector_distances(scan, min_angle, max_angle):
        """Obtiene las distancias válidas dentro de un sector angular."""

        values = []

        for i, distance in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment

            if min_angle <= angle <= max_angle:
                if (
                    math.isfinite(distance)
                    and scan.range_min <= distance <= scan.range_max
                ):
                    values.append(distance)

        return values

    def control_loop(self):
        if self.scan_data is None:
            self.publish_velocity(0.0, 0.0)
            return

        scan = self.scan_data

        # Sector frontal: -25° a +25°
        front = self.sector_distances(
            scan,
            math.radians(-25),
            math.radians(25)
        )

        # Sectores laterales para seleccionar una salida
        left = self.sector_distances(
            scan,
            math.radians(35),
            math.radians(100)
        )

        right = self.sector_distances(
            scan,
            math.radians(-100),
            math.radians(-35)
        )

        # Sin datos frontales válidos, no avanzar a ciegas
        if not front:
            self.publish_velocity(0.0, 0.0)
            return

        front_distance = median(front)

        # Elegir el lado con mayor espacio libre.
        # La dirección queda bloqueada durante la maniobra.
        if not self.avoiding and front_distance < self.safe_distance:
            left_clearance = median(left) if left else 0.0
            right_clearance = median(right) if right else 0.0

            if left_clearance >= right_clearance:
                self.turn_direction = 1.0
            else:
                self.turn_direction = -1.0

            self.avoiding = True
            self.clear_count = 0

            self.get_logger().info(
                f'Obstáculo a {front_distance:.2f} m. '
                f'Girando hacia '
                f'{"izquierda" if self.turn_direction > 0 else "derecha"}.'
            )

        if self.avoiding:
            # Exigir espacio frontal suficiente antes de avanzar.
            if front_distance > self.clear_distance:
                self.clear_count += 1
            else:
                self.clear_count = 0

            # Confirmar que el camino está libre durante varias lecturas.
            if self.clear_count >= 5:
                self.avoiding = False
                self.clear_count = 0

                self.get_logger().info(
                    'Camino frontal despejado. Reanudando avance.'
                )

                self.publish_velocity(self.linear_speed, 0.0)
                return

            # Mantener el sentido de giro evita oscilaciones.
            self.publish_velocity(
                0.0,
                self.turn_direction * self.angular_speed
            )
            return

        # Camino despejado: avanzar lentamente.
        # Reducir velocidad al aproximarse al umbral de seguridad.
        speed_factor = min(
            1.0,
            max(
                0.0,
                (front_distance - self.safe_distance)
                / (self.clear_distance - self.safe_distance)
            )
        )

        speed = self.linear_speed * speed_factor

        self.publish_velocity(speed, 0.0)

    def publish_velocity(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'

        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)

        self.cmd_publisher.publish(msg)

    def stop_robot(self):
        self.publish_velocity(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    navigator = ReactiveNavigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.stop_robot()
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
