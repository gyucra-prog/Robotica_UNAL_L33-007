#!/usr/bin/env python3
"""
Diagnóstico automatizado de ROS 2 + Gazebo + Cartographer.
Entorno objetivo: Ubuntu 24.04, ROS 2 Jazzy, workspace ros2_ws.

Uso:
  python3 diagnostico_cartographer.py
  python3 diagnostico_cartographer.py --launch
  python3 diagnostico_cartographer.py --launch --launch-seconds 25

El modo normal SOLO consulta información. No modifica archivos ni lanza nodos.
El modo --launch intenta reproducir el fallo de Cartographer y luego detiene
el lanzamiento de diagnóstico.
"""

import argparse
import datetime as dt
import difflib
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_WS = Path.home() / "robotica/guia6/ros2_ws"
ERROR_PATTERNS = re.compile(
    r"FATAL|Check failed|ERROR|assertion|terminate called|what\(\)|"
    r"Aborted|SIGABRT|Lookup would require|Extrapolation|"
    r"Could not find|Failed to|not found|No transform",
    re.IGNORECASE,
)
IMPORTANT_PATTERNS = re.compile(
    r"FATAL|Check failed|ERROR|assertion|terminate called|what\(\)|"
    r"Aborted|SIGABRT|Lookup would require|Extrapolation|"
    r"Could not find|Failed to|No transform",
    re.IGNORECASE,
)


class Diagnostic:
    def __init__(self, workspace: Path, launch_mode=False, launch_seconds=25):
        self.workspace = workspace.expanduser().resolve()
        self.package = self.workspace / "src/localization_lab"
        self.launch_mode = launch_mode
        self.launch_seconds = launch_seconds
        self.timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = Path.home() / f"cartographer_diagnostico_{self.timestamp}.txt"
        self.lines = []
        self.findings = []

    def add(self, text=""):
        self.lines.append(str(text))

    def finding(self, level, message):
        self.findings.append((level, message))
        self.add(f"[{level}] {message}")

    def run(self, command, timeout=12, max_chars=12000):
        """Ejecuta comandos ROS dentro de un shell que carga los entornos."""
        ros_setup = Path("/opt/ros/jazzy/setup.bash")
        install_setup = self.workspace / "install/setup.bash"
        command_parts = []
        if ros_setup.exists():
            command_parts.append(f"source {shlex.quote(str(ros_setup))}")
        if install_setup.exists():
            command_parts.append(f"source {shlex.quote(str(install_setup))}")
        command_parts.append(command)
        shell_command = " && ".join(command_parts)

        try:
            result = subprocess.run(
                ["bash", "-lc", shell_command],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            output = result.stdout or ""
            if len(output) > max_chars:
                output = output[:max_chars] + "\n... [salida recortada] ..."
            return result.returncode, output
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            return 124, str(output) + f"\n[TIMEOUT después de {timeout} s]"
        except Exception as exc:
            return 1, f"[No se pudo ejecutar: {exc}]"

    def section_command(self, title, command, timeout=12, max_chars=12000):
        self.add()
        self.add("=" * 78)
        self.add(title)
        self.add("$ " + command)
        code, output = self.run(command, timeout, max_chars)
        self.add(f"[código de salida: {code}]")
        self.add(output.rstrip() if output.strip() else "[sin salida]")
        return code, output

    def check_paths_and_configs(self):
        self.add("\n" + "#" * 78)
        self.add("1. ARCHIVOS Y CONFIGURACIÓN")
        self.add(f"Workspace: {self.workspace}")
        self.add(f"Paquete fuente: {self.package}")

        if not self.workspace.exists():
            self.finding("CRÍTICO", f"No existe el workspace: {self.workspace}")
            return

        source_lua = self.package / "config/cartographer.lua"
        source_launch = self.package / "launch/cartographer.launch.py"
        for label, path in (
            ("cartographer.lua fuente", source_lua),
            ("cartographer.launch.py fuente", source_launch),
            ("setup.py", self.package / "setup.py"),
            ("package.xml", self.package / "package.xml"),
        ):
            status = "OK" if path.is_file() else "NO EXISTE"
            self.add(f"{label}: {status} — {path}")
            if not path.is_file():
                self.finding("ADVERTENCIA", f"No se encontró {label}: {path}")

        code, prefix = self.run("ros2 pkg prefix localization_lab", timeout=10)
        self.add("\nPrefijo instalado del paquete:")
        self.add(prefix.strip() or f"[código {code}]")
        installed_share = None
        if code == 0 and prefix.strip():
            installed_share = Path(prefix.strip().splitlines()[-1]) / "share/localization_lab"

        if installed_share:
            installed_lua = installed_share / "config/cartographer.lua"
            installed_launch = installed_share / "launch/cartographer.launch.py"
            self.add(f"Lua instalado: {installed_lua} ({'OK' if installed_lua.is_file() else 'NO EXISTE'})")
            self.add(f"Launch instalado: {installed_launch} ({'OK' if installed_launch.is_file() else 'NO EXISTE'})")

            if source_lua.is_file() and installed_lua.is_file():
                src_text = source_lua.read_text(errors="replace")
                ins_text = installed_lua.read_text(errors="replace")
                if src_text == ins_text:
                    self.add("Comparación Lua fuente/instalado: IDÉNTICOS")
                else:
                    self.finding("ADVERTENCIA", "cartographer.lua de src y install son diferentes.")
                    self.add("Diferencias (máximo 120 líneas):")
                    diff = list(difflib.unified_diff(
                        src_text.splitlines(), ins_text.splitlines(),
                        fromfile="src/cartographer.lua",
                        tofile="install/cartographer.lua",
                        lineterm=""
                    ))
                    self.add("\n".join(diff[:120]))
            elif source_lua.is_file() and not installed_lua.is_file():
                self.finding("CRÍTICO", "No existe cartographer.lua en la carpeta instalada.")
        return

    def check_ros_graph(self):
        self.add("\n" + "#" * 78)
        self.add("2. ESTADO DE ROS 2 Y GAZEBO")

        commands = [
            ("Distribución ROS", "printenv ROS_DISTRO"),
            ("Nodos activos", "ros2 node list"),
            ("Tópicos relevantes", "ros2 topic list | grep -E '^/(scan|odom|imu|tf|tf_static|clock|map|submap|trajectory)'"),
            ("Información de /scan", "ros2 topic info /scan --verbose"),
            ("Información de /clock", "ros2 topic info /clock"),
            ("Información de /odom", "ros2 topic info /odom"),
            ("Frecuencia de /scan (3 s)", "timeout 3s ros2 topic hz /scan"),
            ("Muestra de /scan", "ros2 topic echo /scan --once"),
            ("Muestra de /clock", "ros2 topic echo /clock --once"),
            ("Procesos Gazebo y bridge", "pgrep -af 'gz sim|gz gui|ros_gz_bridge|robot_state_publisher'"),
        ]
        outputs = {}
        for title, command in commands:
            code, output = self.section_command(title, command, timeout=10, max_chars=5000)
            outputs[title] = output
            if title == "Información de /scan":
                if "Publisher count: 0" in output:
                    self.finding("CRÍTICO", "El tópico /scan no tiene publicador.")
                elif "Publisher count: 1" in output and "Subscription count: 0" in output:
                    self.finding("INFO", "/scan publica datos, pero aún no tiene suscriptores.")
            if title == "Nodos activos":
                if "cartographer_node" not in output:
                    self.add("Nota: cartographer_node no aparece activo en esta consulta.")
        return outputs

    def check_tf(self):
        self.add("\n" + "#" * 78)
        self.add("3. TRANSFORMACIONES TF")
        tf_checks = [
            ("odom -> base_footprint", "timeout 5s ros2 run tf2_ros tf2_echo odom base_footprint"),
            ("base_footprint -> base_link", "timeout 5s ros2 run tf2_ros tf2_echo base_footprint base_link"),
            ("base_link -> base_scan", "timeout 5s ros2 run tf2_ros tf2_echo base_link base_scan"),
            ("map -> odom", "timeout 4s ros2 run tf2_ros tf2_echo map odom"),
        ]
        for title, command in tf_checks:
            code, output = self.section_command(title, command, timeout=8, max_chars=2500)
            has_transform = "Translation:" in output or "Rotation:" in output
            if title != "map -> odom" and not has_transform:
                self.finding("CRÍTICO", f"No se pudo confirmar la transformación {title}.")
            elif title == "map -> odom" and not has_transform:
                self.add("Normal si Cartographer todavía no está ejecutándose.")
        self.section_command(
            "Resumen del árbol TF",
            "ros2 run tf2_tools view_frames",
            timeout=12,
            max_chars=3000
        )

    def check_cartographer_logs(self):
        self.add("\n" + "#" * 78)
        self.add("4. REGISTROS RECIENTES DE ROS 2")

        latest = Path.home() / ".ros/log/latest"
        if not latest.exists():
            self.finding("ADVERTENCIA", "No existe ~/.ros/log/latest.")
            return
        latest = latest.resolve()

        files = []
        try:
            for path in latest.rglob("*"):
                if path.is_file() and path.suffix.lower() in (".log", ".txt", ".out", ".err"):
                    try:
                        if time.time() - path.stat().st_mtime < 24 * 3600:
                            files.append(path)
                    except OSError:
                        pass
        except OSError as exc:
            self.add(f"No se pudieron recorrer los logs: {exc}")
            return

        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        self.add(f"Archivos de log revisados (últimas 24 h): {len(files)}")
        matches = []
        for path in files[:100]:
            try:
                data = path.read_text(errors="replace")[-500_000:]
            except Exception:
                continue
            lines = data.splitlines()
            for index, line in enumerate(lines):
                if ERROR_PATTERNS.search(line):
                    start = max(0, index - 2)
                    end = min(len(lines), index + 3)
                    matches.append((path, index + 1, "\n".join(lines[start:end])))
                    if len(matches) >= 60:
                        break
            if len(matches) >= 60:
                break

        if not matches:
            self.add("No se encontraron patrones de error en los logs recientes.")
        else:
            self.add(f"Coincidencias relevantes: {len(matches)}")
            for path, line_no, excerpt in matches:
                self.add(f"\nArchivo: {path} (línea aproximada {line_no})")
                self.add(excerpt)
            self.finding("REVISAR", "Se encontraron mensajes relevantes en los logs; revisar los extractos anteriores.")

    def reproduce_launch(self):
        self.add("\n" + "#" * 78)
        self.add("5. REPRODUCCIÓN CONTROLADA DEL LANZAMIENTO")

        code, nodes = self.run("ros2 node list", timeout=8)
        if "cartographer_node" in nodes:
            self.finding(
                "OMITIDO",
                "Ya existe un cartographer_node activo. No se lanza otro para evitar duplicados."
            )
            return

        launch_file = self.package / "launch/cartographer.launch.py"
        if not launch_file.is_file():
            self.finding("CRÍTICO", f"No existe el launch fuente: {launch_file}")
            return

        self.add(
            f"Se ejecutará ros2 launch durante un máximo de "
            f"{self.launch_seconds} s. Gazebo debe permanecer activo."
        )
        ros_setup = Path("/opt/ros/jazzy/setup.bash")
        install_setup = self.workspace / "install/setup.bash"
        pieces = []
        if ros_setup.exists():
            pieces.append(f"source {shlex.quote(str(ros_setup))}")
        if install_setup.exists():
            pieces.append(f"source {shlex.quote(str(install_setup))}")
        pieces.append("ros2 launch localization_lab cartographer.launch.py")
        shell_command = " && ".join(pieces)

        process = None
        captured = ""
        try:
            process = subprocess.Popen(
                ["bash", "-lc", shell_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                bufsize=1,
            )
            try:
                captured, _ = process.communicate(timeout=self.launch_seconds)
                self.add(f"El proceso terminó por sí mismo. Código: {process.returncode}")
            except subprocess.TimeoutExpired:
                self.add("Se alcanzó el tiempo de observación; se detiene el lanzamiento de diagnóstico.")
                os.killpg(process.pid, signal.SIGINT)
                try:
                    captured, _ = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    captured, _ = process.communicate(timeout=3)
                self.add(f"Código al detener el lanzamiento: {process.returncode}")
        except Exception as exc:
            self.finding("CRÍTICO", f"No se pudo reproducir el lanzamiento: {exc}")
        finally:
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3)
                except Exception:
                    pass

        captured = captured or ""
        self.add("\n--- SALIDA CAPTURADA DE ROS 2 LAUNCH ---")
        self.add(captured[-30000:] if captured else "[sin salida capturada]")

        error_lines = [
            line for line in captured.splitlines()
            if IMPORTANT_PATTERNS.search(line)
        ]
        if error_lines:
            self.finding("CRÍTICO", "El lanzamiento reprodujo uno o más errores de Cartographer.")
            self.add("\n--- LÍNEAS DE ERROR DETECTADAS ---")
            self.add("\n".join(error_lines[-80:]))
        elif "cartographer_node" in captured and ("started" in captured.lower() or "process started" in captured.lower()):
            self.finding("INFO", "Cartographer parece haber iniciado; revisar nodos y suscripción después del lanzamiento.")
        else:
            self.finding("REVISAR", "No se detectó una firma clara de error; consultar la salida capturada.")

    def write_report(self):
        self.add("\n" + "#" * 78)
        self.add("6. RESUMEN")
        if not self.findings:
            self.add("No se detectaron problemas mediante las comprobaciones automáticas.")
        else:
            for level, message in self.findings:
                self.add(f"[{level}] {message}")

        self.add("\nNotas:")
        self.add("- El script no cambia cartographer.lua, launch.py ni archivos del sistema.")
        self.add("- El modo normal no inicia ni detiene nodos.")
        self.add("- Las comprobaciones TF pueden mostrar errores esperables para map -> odom si Cartographer no está activo.")
        self.add("- Un RTF bajo de Gazebo no implica por sí solo un error de Cartographer.")

        self.report_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print("\n".join(self.lines[-35:]))
        print("\n" + "=" * 78)
        print(f"INFORME COMPLETO: {self.report_path}")
        print("=" * 78)

    def run_all(self):
        self.add("DIAGNÓSTICO AUTOMATIZADO CARTOGRAPHER / ROS 2 JAZZY")
        self.add(f"Fecha: {dt.datetime.now().isoformat(timespec='seconds')}")
        self.add(f"Usuario: {os.environ.get('USER', 'desconocido')}")
        self.add(f"Workspace: {self.workspace}")
        self.add(f"Modo: {'diagnóstico + reproducción' if self.launch_mode else 'solo diagnóstico'}")

        self.check_paths_and_configs()
        self.check_ros_graph()
        self.check_tf()
        self.check_cartographer_logs()
        if self.launch_mode:
            self.reproduce_launch()
        self.write_report()


def main():
    parser = argparse.ArgumentParser(
        description="Reúne en un solo informe los diagnósticos de Cartographer."
    )
    parser.add_argument(
        "--workspace",
        default=str(DEFAULT_WS),
        help=f"Ruta del workspace (predeterminada: {DEFAULT_WS})"
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Intenta iniciar Cartographer para capturar el error. Gazebo debe estar activo."
    )
    parser.add_argument(
        "--launch-seconds",
        type=int,
        default=25,
        help="Segundos máximos de observación en modo --launch (predeterminado: 25)."
    )
    args = parser.parse_args()

    if args.launch_seconds < 5 or args.launch_seconds > 120:
        parser.error("--launch-seconds debe estar entre 5 y 120.")

    diagnostic = Diagnostic(
        Path(args.workspace),
        launch_mode=args.launch,
        launch_seconds=args.launch_seconds
    )
    diagnostic.run_all()


if __name__ == "__main__":
    main()
