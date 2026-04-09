import sys
if sys.prefix == '/home/gsmst/.conda/envs/dogenv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/gsmst/gsmst/Unitree-Go2-Control/ros2_ws/install/lidar_processor'
