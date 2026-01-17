"""
Defines DDS topic names used internally by the system
for inter-process communication. These topics are used by ROS2 bridges,
LIDAR processing, and state monitoring.

Students should **not** use these directly; they are for internal
infrastructure and low-level module integration only.

DDS_TOPICS : dict
    Mapping of topic identifiers to DDS topic strings:

    - "LOW_STATE" : Topic for low-level robot state updates
    - "RAW_LIDAR" : Topic for raw, deskewed LiDAR point clouds
    - "ODOMETRY_LIDAR" : Topic for LiDAR point clouds used in odometry
    - "HEIGHT_MAP_ARRAY" : Topic for LiDAR height map arrays
    - "RANGE_INFO" : Topic for LiDAR range information
"""

DDS_TOPICS = {
    "LOW_STATE": "rt/lf/lowstate",
    "RAW_LIDAR": "rt/utlidar/cloud_deskewed",
    "ODOMETRY_LIDAR": "rt/utlidar/cloud",
    "HEIGHT_MAP_ARRAY": "rt/utlidar/height_map_array",
    "RANGE_INFO": "rt/utlidar/range_info"
}