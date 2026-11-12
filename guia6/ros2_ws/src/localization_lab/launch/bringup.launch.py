from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # --------------------------------------------------
    # CONFIGURACION GENERAL
    # --------------------------------------------------

    use_sim_time = LaunchConfiguration('use_sim_time')

    # Directorios de los paquetes
    gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    localization_dir = get_package_share_directory('localization_lab')
    nav2_dir = get_package_share_directory('nav2_bringup')

    # Archivos de lanzamiento
    gazebo_launch_file = os.path.join(
        gazebo_dir,
        'launch',
        'turtlebot3_house.launch.py'
    )

    cartographer_launch_file = os.path.join(
        localization_dir,
        'launch',
        'cartographer.launch.py'
    )

    navigation_launch_file = os.path.join(
        localization_dir,
        'launch',
        'navigation.launch.py'
    )

    # Configuracion de RViz2
    rviz_config = os.path.join(
        nav2_dir,
        'rviz',
        'nav2_default_view.rviz'
    )

    # --------------------------------------------------
    # 1. GAZEBO
    # --------------------------------------------------

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file)
    )

    # --------------------------------------------------
    # 2. CARTOGRAPHER SLAM
    # Se inicia despues de Gazebo
    # --------------------------------------------------

    cartographer = TimerAction(
        period=25.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    cartographer_launch_file
                )
            )
        ]
    )

    # --------------------------------------------------
    # 3. NAV2
    # Se inicia despues de Cartographer
    # --------------------------------------------------

    navigation = TimerAction(
        period=40.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    navigation_launch_file
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time
                }.items()
            )
        ]
    )

    # --------------------------------------------------
    # 4. RVIZ2
    # --------------------------------------------------

    rviz = TimerAction(
        period=50.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=[
                    '-d',
                    rviz_config
                ],
                parameters=[
                    {'use_sim_time': use_sim_time}
                ]
            )
        ]
    )

    # --------------------------------------------------
    # LANZAMIENTO MAESTRO
    # --------------------------------------------------

    return LaunchDescription([

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Usar reloj simulado de Gazebo'
        ),

        SetEnvironmentVariable(
            name='TURTLEBOT3_MODEL',
            value='burger'
        ),

        gazebo,
        cartographer,
        navigation,
        rviz

    ])
