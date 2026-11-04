from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    config_dir = os.path.join(
        get_package_share_directory('localization_lab'),
        'config'
    )

    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '-configuration_directory',
            config_dir,
            '-configuration_basename',
            'cartographer.lua'
        ]
    )

    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '-resolution',
            '0.05'
        ]
    )

    return LaunchDescription([
        cartographer_node,
        occupancy_grid_node
    ])
