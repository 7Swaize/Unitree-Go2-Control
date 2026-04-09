import os
import yaml
import numpy as np
import fast_pointcloud as fp

from ament_index_python import get_package_share_directory
from lidar_processor.lidar_filter_node import FilterConfig


yaml_path = os.path.join(get_package_share_directory('bringup'), 'config', 'lidar_processor.yaml')
with open(yaml_path, 'r') as f:
    config_data = yaml.safe_load(f)

config_params = config_data['lidar_filter']['ros__parameters']['filter']


num_points = 1500000
xyz_data = np.random.default_rng().uniform(low=-5, high=25, size=(num_points, 3))
intensity_data = np.random.default_rng().uniform(low=-50, high=200, size=(num_points, 1))
config = FilterConfig(**config_params)

fp.apply_filter(np.hstack([xyz_data, intensity_data]), config)



# py-spy record -o profile.svg -- python src/lidar_processor/test/profiling/filter_profiling.py
# py-spy top -- python src/lidar_processor/test/profiling/filter_profiling.py
# 