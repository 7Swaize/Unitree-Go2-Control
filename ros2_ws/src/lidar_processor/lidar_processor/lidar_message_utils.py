from typing import Tuple, Optional
import numpy as np
from sensor_msgs.msg import PointField

import fast_pointcloud as fp
from go2_interfaces.msg import LidarDecoded


NP_DTYPE_TO_CODE = {
    np.int8: 1,
    np.uint8: 2,
    np.int16: 3,
    np.uint16: 4,
    np.int32: 5,
    np.uint32: 6,
    np.float32: 7,
    np.float64: 8
}

CODE_TO_NP_DTYPE = {v: k for k, v in NP_DTYPE_TO_CODE.items()}

POINTFIELD_TO_INTERNAL_CTYPE = {
    PointField.INT8: fp.PointFieldType["INT8"],
    PointField.UINT8: fp.PointFieldType["UINT8"],
    PointField.INT16: fp.PointFieldType["INT16"],
    PointField.UINT16: fp.PointFieldType["UINT16"],
    PointField.INT32: fp.PointFieldType["INT32"],
    PointField.UINT32: fp.PointFieldType["UINT32"],
    PointField.FLOAT32: fp.PointFieldType["FLOAT32"],
    PointField.FLOAT64: fp.PointFieldType["FLOAT64"]
}


def encode_array_to_message(array: np.ndarray, arr_prefix: str) -> dict:
    return {
        f"{arr_prefix}_shape": list(array.shape),
        f"{arr_prefix}_dtype": NP_DTYPE_TO_CODE[array.dtype],
        f"{arr_prefix}_data": array.tobytes(order='C')
    }


def decode_array_from_message(msg: LidarDecoded, arr_prefix: str) -> np.ndarray:
    shape_attr = f"{arr_prefix}_shape"
    dtype_attr = f"{arr_prefix}_dtype"
    data_attr = f"{arr_prefix}_data"
    
    shape = tuple(getattr(msg, shape_attr))
    dtype = CODE_TO_NP_DTYPE[getattr(msg, dtype_attr)]
    data = getattr(msg, data_attr)
    
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def create_lidar_decoded_message(xyz: np.ndarray, intensity: Optional[np.ndarray] = None) -> LidarDecoded:
    msg = LidarDecoded()
    
    xyz_encoded = encode_array_to_message(xyz, "xyz")
    msg.xyz_shape = xyz_encoded["xyz_shape"]
    msg.xyz_dtype = xyz_encoded["xyz_dtype"]
    msg.xyz_data = xyz_encoded["xyz_data"]
    
    if intensity is not None:
        msg.has_intensity = True
        intensity_encoded = encode_array_to_message(intensity, "intensity")
        msg.intensity_shape = intensity_encoded["intensity_shape"]
        msg.intensity_dtype = intensity_encoded["intensity_dtype"]
        msg.intensity_data = intensity_encoded["intensity_data"]
    else:
        msg.has_intensity = False
        msg.intensity_shape = []
        msg.intensity_dtype = 0
        msg.intensity_data = []
    
    return msg
