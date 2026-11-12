#!/usr/bin/env python3

import time
import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy
)

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


class DiagnosticoRobot(Node):

    def __init__(self):
        super().__init__('diagnostico_robot')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.datos = {}

        topicos = {
            '/scan': LaserScan,
            '/odom': Odometry,
            '/cmd_vel_nav': TwistStamped,
            '/cmd_vel_smoothed': TwistStamped,
            '/cmd_vel': TwistStamped
        }

        for nombre, tipo in topicos.items():
            self.datos[nombre] = {
                'contador': 0,
                'ultimo_tiempo': None,
                'frecuencia': 0.0,
                'mensaje': None
            }

            self.create_subscription(
                tipo,
                nombre,
                lambda msg, topic=nombre: self.callback(msg, topic),
                qos
            )

        self.timer = self.create_timer(2.0, self.informe)

        self.get_logger().info(
            'Monitor iniciado. Esperando datos ROS 2...'
        )

    def callback(self, msg, topic):

        ahora = time.monotonic()
        dato = self.datos[topic]

        if dato['ultimo_tiempo'] is not None:
            dt = ahora - dato['ultimo_tiempo']

            if dt > 0:
                frecuencia_actual = 1.0 / dt

                if dato['frecuencia'] == 0:
                    dato['frecuencia'] = frecuencia_actual
                else:
                    dato['frecuencia'] = (
                        0.7 * dato['frecuencia']
                        + 0.3 * frecuencia_actual
                    )

        dato['ultimo_tiempo'] = ahora
        dato['contador'] += 1
        dato['mensaje'] = msg

    def informe(self):

        ahora = time.monotonic()

        print('\n' + '=' * 65)
        print('          DIAGNOSTICO DEL TURTLEBOT3')
        print('=' * 65)

        for topic, dato in self.datos.items():

            ultimo = dato['ultimo_tiempo']

            if ultimo is None:
                print(f'{topic:22} SIN DATOS')
                continue

            antiguedad = ahora - ultimo

            print(
                f'{topic:22} '
                f'{dato["frecuencia"]:7.2f} Hz | '
                f'último mensaje: {antiguedad:.2f} s'
            )

            if topic == '/scan':
                msg = dato['mensaje']

                print(
                    f'  Rangos: {len(msg.ranges)} | '
                    f'mínimo: {msg.range_min:.2f} m | '
                    f'máximo: {msg.range_max:.2f} m'
                )

            elif topic == '/odom':
                msg = dato['mensaje']

                v = msg.twist.twist.linear.x
                w = msg.twist.twist.angular.z

                print(
                    f'  Odometría: v={v:.3f} m/s | '
                    f'w={w:.3f} rad/s'
                )

            elif topic.startswith('/cmd_vel'):

                msg = dato['mensaje']
                v = msg.twist.linear.x
                w = msg.twist.angular.z

                print(
                    f'  Comando: v={v:.3f} m/s | '
                    f'w={w:.3f} rad/s'
                )

                if abs(v) < 0.001 and abs(w) < 0.001:
                    print('  ESTADO: PARADA')

                elif abs(w) > 1.0 and abs(v) > 0.05:
                    print('  ALERTA: GIRO PRONUNCIADO')

                elif abs(v) > 0.05:
                    print('  ESTADO: AVANCE')

        print('=' * 65)


def main(args=None):

    rclpy.init(args=args)
    nodo = DiagnosticoRobot()

    try:
        rclpy.spin(nodo)

    except KeyboardInterrupt:
        print('\nMonitor detenido.')

    finally:
        nodo.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
