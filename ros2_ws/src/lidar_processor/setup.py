from setuptools import find_packages, setup

package_name = 'lidar_processor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gsmst',
    maintainer_email='rahejasachit@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lidar_decoder_node = lidar_processor.lidar_decoder_node:main',
            'lidar_filter_node = lidar_processor.lidar_filter_node:main',
            'ros_bridge_node = lidar_processor.ros_bridge_node:main'  
        ],
    },
)



'''
ROS2 packages needed by conda:
pip install -U pip setuptools wheel
pip install colcon-common-extensions
pip install empy catkin_pkg lark-parser

then:
export PYTHON_EXECUTABLE=$(which python)
export AMENT_PYTHON_EXECUTABLE=$(which python)
source /opt/ros/humble/setup.bash
'''