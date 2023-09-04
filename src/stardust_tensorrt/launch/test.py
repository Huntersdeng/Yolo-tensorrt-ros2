import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('stardust_tensorrt')
    paramer_dir = os.path.join(pkg_path, 'config', 'cfg.yaml')
    return LaunchDescription([
        Node(
            package='stardust_tensorrt', 
            executable='yolo',
            name='yolo',
            parameters=[paramer_dir],
            arguments=["--ros-args", "--log-level", "info"],
            output='screen')
    ])
