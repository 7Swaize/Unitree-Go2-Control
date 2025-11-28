from array import array
from collections import namedtuple
import collections
import math
import struct
import threading
from typing import Any, Generator, List, Optional, Tuple

import cyclonedds.core as dds
import open3d as o3d

import numpy as np
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_


DOMAIN_ID = 0

# downsample params
SCAN_DOWNSAMPLE = 0.15   # for feature extraction / ICP input
MAP_VOXEL = 0.06         # voxel size for fused global map
KEYFRAME_TRANSLATION = 0.5  # meters: create a new keyframe after this much movement
KEYFRAME_INTERVAL = 1.0     # seconds minimum between keyframes
MIN_POINTS_KEYFRAME = 200   # minimum points to accept a keyframe

# registration params
ICP_MAX_DIST = 1.0
ICP_ITER = 50

# FPFH (for loop detection)
FPFH_RADIUS_NORMAL = 0.5
FPFH_RADIUS_FEATURE = 1.0
RANSAC_DISTANCE_THRESHOLD = 1.5
RANSAC_NUM_ITER = 400000
RANSAC_CONFIDENCE = 0.999

# loop closure scanning window and frequency
LOOP_SEARCH_EVERY_N_KEYFRAMES = 5
LOOP_SEARCH_RADIUS = 10.0  # meters: only search earlier keyframes within this distance

# optimization
OPTIMIZE_EVERY_N_KEYFRAMES = 10


_DATATYPES = {
    1: ('b', 1),  # INT8
    2: ('B', 1),  # UINT8
    3: ('h', 2),  # INT16
    4: ('H', 2),  # UINT16
    5: ('i', 4),  # INT32
    6: ('I', 4),  # UINT32
    7: ('f', 4),  # FLOAT32
    8: ('d', 8),  # FLOAT64
}

def _get_struct_fmt(is_bigendian: bool, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in sorted(fields, key=lambda f: int(f.offset)):
        if field_names is not None and field.name not in field_names:
            continue

        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset 

        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype)
            continue

        datatype_fmt, datatype_length = _DATATYPES[field.datatype]
        fmt += datatype_fmt * max(1, getattr(field, 'count', 1))
        offset += datatype_length * max(1, getattr(field, 'count', 1))

    return fmt


def _read_points(cloud: PointCloud2_, field_names=None, skip_nans=True) -> Generator[Tuple[Any, ...], None, None]:
    """
    Generator: returns a tuple (values...) for each point, fields appear in order of field_names.
    If field_names is None, returns all fields in message order.
    """
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    # Possibly check forced conversion into buffer using data_buf = memoryview(list(cloud.data))
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, bytes(cloud.data), math.isnan # type: ignore
    unpack_from = struct.Struct(fmt).unpack_from
    

    if skip_nans:
        for v in range(height):
            offset = row_step * v
            for u in range(width):
                p = unpack_from(bytearray(data), offset)
                
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
                offset += point_step
    
    else:
        for v in range(height):
            offset = row_step * v
            for u in range(width):
                yield unpack_from(data, offset)
                offset += point_step


class PointCloudDecoder:
    def __init__(self) -> None:
        pass

    def cloud_to_numpy(self, cloud: PointCloud2_) -> np.ndarray:
        points = []
        use_fields = ['x', 'y', 'z', 'intensity'] # assuming there is intensity field?

        for p in _read_points(cloud, use_fields):
            points.append(p)

        return np.array(points, dtype=np.float32)


class HeightMapDecoder:
    def __init__(self) -> None:
        pass

    def heightmap_to_numpy(self, heightmap: HeightMap_):
        ox, oy = heightmap.origin[0], heightmap.origin[1]  # type: ignore
        res = heightmap.resolution

        points = []
        for i in range(heightmap.height):
            for j in range(heightmap.width):
                h = heightmap.data[i * heightmap.width + j] # type: ignore
                x = ox + j * res
                y = oy + i * res
                z = h

                points.append([x, y, z])

        points = np.array(points, dtype=np.float32)


Keyframe = collections.namedtuple('Keyframe', ['idx', 'time', 'pose', 'pcd', 'pcd_down', 'fpfh']) 

# RANSAC + FPFH https://medium.com/@amnahhmohammed/gentle-introduction-to-point-cloud-registration-using-open3d-pt-2-18df4cb8b16c
# ICP: https://www.youtube.com/watch?v=4uWSo8v3iQA
class PoseGraphSLAM:
    def __init__(self) -> None:
        self.keyframes: List[Keyframe] = []
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.optimized = False
    

    def add_keyframe(self, kf: Keyframe, edge_to_prev:Optional[o3d.pipelines.registration.PoseGraphEdge]=None):
        if not self.keyframes:
            # add first node at identity
            node = o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(kf.pose))
            self.pose_graph.nodes.append(node)
        else:
            node = o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(kf.pose))
            self.pose_graph.nodes.append(node)
            if edge_to_prev is not None:
                self.pose_graph.edges.append(edge_to_prev)

        self.keyframes.append(kf)


    def add_loop_edge(self, i_from:int, i_to:int, trans:np.ndarray, information=np.eye(6)*100):
        # Add a loop closure as an uncertain constraint (set uncertain=False if you trust it)
        edge = o3d.pipelines.registration.PoseGraphEdge(i_from, i_to, trans, information, uncertain=False)
        self.pose_graph.edges.append(edge)
    

    def optimize(self):
        # global optimization settings
        option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=ICP_MAX_DIST,
                                                                     edge_prune_threshold=0.25,
                                                                     reference_node=0)
        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        o3d.pipelines.registration.global_optimization(self.pose_graph,
                                                      method,
                                                      o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                      option)
        self.optimized = True


# -----------------------------
# Feature helpers (FPFH)
# -----------------------------
def estimate_normals(pcd, radius):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

def compute_fpfh(pcd, radius):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )


# -----------------------------
# Registration helpers
# -----------------------------
def run_icp(source_down, target_down, init=np.eye(4), max_dist=ICP_MAX_DIST):
    # ensure normals
    if not target_down.has_normals():
        estimate_normals(target_down, FPFH_RADIUS_NORMAL)
    if not source_down.has_normals():
        estimate_normals(source_down, FPFH_RADIUS_NORMAL)
    reg = o3d.pipelines.registration.registration_icp(
        source_down, target_down, max_dist, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_ITER)
    )

    return reg

def ransac_initial_alignment(source_down, target_down, source_fpfh, target_fpfh):
    # Use RANSAC to get coarse transform candidate
    distance = RANSAC_DISTANCE_THRESHOLD
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_NUM_ITER, 500)
    )

    return result


class LiDARSLAM:
    def __init__(self):
        self.pg = PoseGraphSLAM()
        self.last_keyframe_pose = np.eye(4)
        self.last_keyframe_time = 0.0
        self.keyframe_idx = 0
        self.lock = threading.Lock()

        self.vis = o3d.
        self.vis.create_window("GO2 LiDAR SLAM", width=1400, height=900)
        self.geom_added = False