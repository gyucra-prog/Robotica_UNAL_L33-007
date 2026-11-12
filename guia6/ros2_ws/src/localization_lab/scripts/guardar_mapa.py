#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy
)

import numpy as np
from pathlib import Path
import math


class GuardarMapa(Node):

    def __init__(self):
        super().__init__('guardar_mapa')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.callback_mapa,
            qos
        )

        self.guardado = False

        self.get_logger().info(
            'Esperando mapa de Cartographer en /map...'
        )

    def callback_mapa(self, msg):

        if self.guardado:
            return

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        if width == 0 or height == 0:
            self.get_logger().error('El mapa está vacío.')
            return

        self.get_logger().info(
            f'Mapa recibido: {width} x {height} celdas'
        )

        self.get_logger().info(
            f'Resolución: {resolution:.3f} m/celda'
        )

        data = np.array(
            msg.data,
            dtype=np.int16
        ).reshape((height, width))

        # Convertir ocupación a escala de grises.
        # Libre = blanco, ocupado = negro, desconocido = gris.

        imagen = np.full(
            (height, width),
            205,
            dtype=np.uint8
        )

        imagen[data == 0] = 254
        imagen[data >= 65] = 0

        # Invertir verticalmente para guardar correctamente
        # la orientación de la imagen PGM.

        imagen = np.flipud(imagen)

        carpeta = Path.home() / 'robotica' / 'mapas'
        carpeta.mkdir(parents=True, exist_ok=True)

        ruta_pgm = carpeta / 'mapa_robot.pgm'
        ruta_yaml = carpeta / 'mapa_robot.yaml'

        # Guardar imagen PGM.
        with open(ruta_pgm, 'wb') as archivo:
            archivo.write(
                f'P5\n{width} {height}\n255\n'.encode()
            )
            archivo.write(imagen.tobytes())

        # Obtener orientación del origen.
        q = msg.info.origin.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )

        origen = msg.info.origin.position

        contenido_yaml = f"""image: {ruta_pgm.name}
mode: trinary
resolution: {resolution}
origin: [{origen.x}, {origen.y}, {yaw}]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""

        with open(ruta_yaml, 'w') as archivo:
            archivo.write(contenido_yaml)

        self.guardado = True

        self.get_logger().info(
            f'Mapa guardado: {ruta_pgm}'
        )

        self.get_logger().info(
            f'Metadatos guardados: {ruta_yaml}'
        )

        self.get_logger().info(
            'Captura completada correctamente.'
        )

        raise SystemExit(0)


def main(args=None):

    rclpy.init(args=args)

    nodo = GuardarMapa()

    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        nodo.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
