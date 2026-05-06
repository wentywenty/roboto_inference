##launch file
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory("roboto-inference")

    inference_config_arg = DeclareLaunchArgument(
        "inference_config",
        default_value="inference.yaml",
        description="Inference config yaml file path",
    )

    robot_config_arg = DeclareLaunchArgument(
        "robot_config",
        default_value="robot.yaml",
        description="Robot config yaml file path",
    )

    return LaunchDescription(
        [
            inference_config_arg,
            robot_config_arg,
            Node(
                package="joy",
                executable="joy_node",
                name="joy_node",
                output="screen",
            ),
            Node(
                package="roboto-inference",
                executable="inference_node",
                name="inference_node",
                parameters=[
                    LaunchConfiguration("inference_config"),
                    {"robot_config": LaunchConfiguration("robot_config")},
                ],
                output="screen",
                # prefix=["xterm -e gdb -ex run --args"],
            ),
        ]
    )
