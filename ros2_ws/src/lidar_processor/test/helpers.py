import os
from ament_index_python.packages import get_package_share_directory

def start_decoder_node(self):
    params_file = os.path.join(get_package_share_directory('bringup'), 'config', 'lidar_processor.yaml')