import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)

import tf2_ros


class SimpleController(Node):

    def __init__(self):
        super().__init__('simple_controller')

        # Parámetros de movimiento
        self.linear_speed = 0.12
        self.angular_speed = 0.8
        self.obstacle_distance = 0.55
        self.goal_tolerance = 0.25

        # Datos de los sensores
        self.scan = None
        self.frontiers = []

        # Publicador de velocidad
        self.publisher_ = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        # Suscriptor LiDAR
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # Suscriptor del mapa
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        # Transformaciones TF
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=Duration(seconds=10.0)
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # Temporizador de control: 10 Hz
        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            'Controlador de exploracion iniciado'
        )

    # -------------------------------------------------
    # RECEPCION DEL LIDAR
    # -------------------------------------------------

    def scan_callback(self, msg):
        self.scan = msg

    # -------------------------------------------------
    # RECEPCION Y PROCESAMIENTO DEL MAPA
    # -------------------------------------------------

    def map_callback(self, msg):

        width = msg.info.width
        height = msg.info.height
        data = msg.data

        frontiers = []

        # Buscar celdas libres junto a celdas desconocidas
        for y in range(1, height - 1):
            for x in range(1, width - 1):

                index = y * width + x

                if data[index] < 0 or data[index] >= 50:
                    continue

                neighbors = [
                    data[index - 1],
                    data[index + 1],
                    data[index - width],
                    data[index + width]
                ]

                if any(value == -1 for value in neighbors):

                    # Centro de la celda en coordenadas del mapa
                    local_x = (x + 0.5) * msg.info.resolution
                    local_y = (y + 0.5) * msg.info.resolution

                    origin = msg.info.origin
                    q = origin.orientation

                    origin_yaw = math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    )

                    world_x = (
                        origin.position.x
                        + local_x * math.cos(origin_yaw)
                        - local_y * math.sin(origin_yaw)
                    )

                    world_y = (
                        origin.position.y
                        + local_x * math.sin(origin_yaw)
                        + local_y * math.cos(origin_yaw)
                    )

                    frontiers.append((world_x, world_y))

        self.frontiers = frontiers

    # -------------------------------------------------
    # OBTENER POSICION DEL ROBOT EN EL MAPA
    # -------------------------------------------------

    def get_robot_pose(self):

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                Time()
            )

            position = transform.transform.translation
            rotation = transform.transform.rotation

            yaw = math.atan2(
                2.0 * (
                    rotation.w * rotation.z
                    + rotation.x * rotation.y
                ),
                1.0 - 2.0 * (
                    rotation.y ** 2
                    + rotation.z ** 2
                )
            )

            return position.x, position.y, yaw

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ):
            return None

    # -------------------------------------------------
    # ANALISIS DE OBSTACULOS
    # -------------------------------------------------

    def analyze_scan(self):

        if self.scan is None:
            return None

        sectors = {
            'front': [],
            'left': [],
            'right': []
        }

        scan = self.scan

        for i, distance in enumerate(scan.ranges):

            angle = scan.angle_min + i * scan.angle_increment

            # Ignorar lecturas invalidas
            if not math.isfinite(distance):
                continue

            if distance < scan.range_min or distance > scan.range_max:
                continue

            angle = math.atan2(math.sin(angle), math.cos(angle))

            if abs(angle) <= math.radians(30):
                sectors['front'].append(distance)

            elif math.radians(30) < angle <= math.radians(100):
                sectors['left'].append(distance)

            elif -math.radians(100) <= angle < -math.radians(30):
                sectors['right'].append(distance)

        result = {}

        for name, values in sectors.items():
            result[name] = min(values) if values else scan.range_max

        return result

    # -------------------------------------------------
    # PUBLICAR VELOCIDAD
    # -------------------------------------------------

    def publish_velocity(self, linear, angular):

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)

        self.publisher_.publish(msg)

    # -------------------------------------------------
    # CONTROL PRINCIPAL
    # -------------------------------------------------

    def control_loop(self):

        if self.scan is None or not self.frontiers:
            self.publish_velocity(0.0, 0.0)
            return

        pose = self.get_robot_pose()

        if pose is None:
            self.publish_velocity(0.0, 0.0)
            return

        x, y, yaw = pose

        obstacles = self.analyze_scan()

        if obstacles is None:
            self.publish_velocity(0.0, 0.0)
            return

        # Prioridad 1: evitar obstaculos
        if obstacles['front'] < self.obstacle_distance:

            self.get_logger().info(
                'Obstaculo detectado: buscando espacio libre',
                throttle_duration_sec=2.0
            )

            if obstacles['left'] >= obstacles['right']:
                direction = 1.0
            else:
                direction = -1.0

            self.publish_velocity(0.0, direction * self.angular_speed)
            return

        # Prioridad 2: seleccionar la frontera mas cercana
        target = min(
            self.frontiers,
            key=lambda point: (
                (point[0] - x) ** 2
                + (point[1] - y) ** 2
            )
        )

        target_x, target_y = target

        distance = math.hypot(
            target_x - x,
            target_y - y
        )

        if distance < self.goal_tolerance:
            self.publish_velocity(0.0, 0.0)
            return

        # Orientacion hacia la frontera
        target_angle = math.atan2(
            target_y - y,
            target_x - x
        )

        angle_error = math.atan2(
            math.sin(target_angle - yaw),
            math.cos(target_angle - yaw)
        )

        # Control proporcional de orientacion
        angular = 1.5 * angle_error
        angular = max(
            -self.angular_speed,
            min(self.angular_speed, angular)
        )

        # Reducir velocidad si el robot necesita girar
        if abs(angle_error) > math.radians(45):
            linear = 0.0
        else:
            linear = self.linear_speed * (
                1.0 - abs(angle_error) / math.radians(90)
            )

        self.publish_velocity(linear, angular)


def main(args=None):

    rclpy.init(args=args)

    node = SimpleController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.publish_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
