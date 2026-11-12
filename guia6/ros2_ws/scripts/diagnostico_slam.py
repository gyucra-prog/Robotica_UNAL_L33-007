#!/usr/bin/env python3

import rclpy
import time
from pathlib import Path

from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException


# ==========================================================
# CONFIGURACION
# ==========================================================

CONFIG_DIR = (
    Path.home()
    / "robotica/guia6/ros2_ws/src/localization_lab/config"
)

ARCHIVOS_LUA = [
    "cartographer.lua",
    "map_builder.lua",
    "trajectory_builder.lua",
    "pose_graph.lua",
    "trajectory_builder_2d.lua",
    "trajectory_builder_3d.lua",
]

TRANSFORMACIONES = [
    ("odom", "base_footprint"),
    ("base_footprint", "base_scan"),
    ("base_footprint", "base_link"),
]


# ==========================================================
# DIAGNOSTICO DE ARCHIVOS
# ==========================================================

def verificar_archivos():

    print("\n" + "=" * 60)
    print("  1. VERIFICACION DE ARCHIVOS LUA")
    print("=" * 60)

    if not CONFIG_DIR.exists():
        print("[ERROR] No existe el directorio:")
        print(CONFIG_DIR)
        return False

    total_bytes = 0
    faltantes = []

    for nombre in ARCHIVOS_LUA:

        archivo = CONFIG_DIR / nombre

        if archivo.is_file():

            size = archivo.stat().st_size
            total_bytes += size

            print(
                f"[OK] {nombre:<30} "
                f"{size:>6} bytes"
            )

        else:

            faltantes.append(nombre)
            print(f"[FALTA] {nombre}")

    print("-" * 60)

    print(f"Total real: {total_bytes} bytes")
    print(f"Total aproximado: {total_bytes / 1024:.2f} KB")
    print(f"Archivos faltantes: {len(faltantes)}")

    return len(faltantes) == 0


# ==========================================================
# NODO DE DIAGNOSTICO TF
# ==========================================================

class DiagnosticoTF(Node):

    def __init__(self):

        super().__init__("diagnostico_slam_tf")

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

    def consultar_transformacion(
        self,
        target_frame,
        source_frame,
        timeout=5.0
    ):

        print("\n" + "-" * 60)

        print(
            f"TRANSFORMACION: "
            f"{target_frame} <- {source_frame}"
        )

        inicio = time.monotonic()

        while time.monotonic() - inicio < timeout:

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            try:

                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time()
                )

                t = transform.transform.translation
                q = transform.transform.rotation

                print("[OK] Transformacion disponible")

                print("\nTraslacion:")
                print(f"  X: {t.x:.4f} m")
                print(f"  Y: {t.y:.4f} m")
                print(f"  Z: {t.z:.4f} m")

                print("\nOrientacion (cuaternion):")
                print(f"  X: {q.x:.5f}")
                print(f"  Y: {q.y:.5f}")
                print(f"  Z: {q.z:.5f}")
                print(f"  W: {q.w:.5f}")

                stamp = transform.header.stamp

                print(
                    f"\nMarca temporal: "
                    f"{stamp.sec}.{stamp.nanosec:09d}"
                )

                return True

            except TransformException:
                continue

        print("[ERROR] Transformacion no disponible")
        print(
            f"No se encontro {target_frame} <- "
            f"{source_frame} en {timeout} segundos."
        )

        return False


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

def main():

    print("\n")
    print("=" * 60)
    print("       DIAGNOSTICO DEL SISTEMA SLAM - ROS 2")
    print("=" * 60)

    # 1. Verificar archivos

    archivos_ok = verificar_archivos()

    # 2. Inicializar ROS 2

    print("\n" + "=" * 60)
    print("  2. VERIFICACION DE TRANSFORMACIONES TF")
    print("=" * 60)

    rclpy.init()

    nodo = DiagnosticoTF()

    resultados = {}

    try:

        for target, source in TRANSFORMACIONES:

            clave = f"{target} <- {source}"

            resultados[clave] = (
                nodo.consultar_transformacion(
                    target,
                    source,
                    timeout=5.0
                )
            )

    except KeyboardInterrupt:

        print("\nDiagnostico interrumpido.")

    finally:

        nodo.destroy_node()
        rclpy.shutdown()

    # 3. Resumen

    print("\n" + "=" * 60)
    print("  3. RESUMEN DEL DIAGNOSTICO")
    print("=" * 60)

    print(
        "Archivos Lua:",
        "CORRECTOS" if archivos_ok else "INCOMPLETOS"
    )

    for nombre, estado in resultados.items():

        print(
            f"{nombre:<40} "
            f"{'OK' if estado else 'NO DISPONIBLE'}"
        )

    print("=" * 60)
    print("Diagnostico finalizado.")
    print("=" * 60)


if __name__ == "__main__":
    main()
