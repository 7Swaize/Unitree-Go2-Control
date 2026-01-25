# fast_pointcloud.pyi
from typing import Tuple, Optional
import numpy

# PointField data types
PF_INT8: int
PF_UINT8: int
PF_INT16: int
PF_UINT16: int
PF_INT32: int
PF_UINT32: int
PF_FLOAT32: int
PF_FLOAT64: int


def decode_xyz_intensity(
    data: bytes,
    point_step: int,
    ox: int,
    oy: int,
    oz: int,
    oi: int,
    is_bigendian: int,
    dtype_xyz: int,
    dtype_intensity: int,
    skip_nans: int
) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
    """
    Decode XYZ (+ optional intensity) from a PointCloud2 byte buffer.

    Parameters
    ----------
    data : bytes
        Raw point cloud data.
    point_step : int
        Size of one point in bytes.
    ox, oy, oz : int
        Offsets of x, y, z fields in bytes.
    oi : int
        Offset of intensity field (-1 if none).
    is_bigendian : int
        Endianness flag (1 = big-endian, 0 = little-endian).
    dtype_xyz : int
        Data type of XYZ fields (PF_* constants).
    dtype_intensity : int
        Data type of intensity field (PF_* constants).
    skip_nans : int
        Whether to skip points with NaNs.

    Returns
    -------
    Tuple[numpy.ndarray, Optional[numpy.ndarray]]
        xyz : ndarray of shape (n_points, 3), dtype float64
        intensity : ndarray of shape (n_points,) or None
    """
    ...


class FilterConfig:
    """
    Configuration for point cloud filtering.

    Attributes
    ----------
    max_range : float
    min_range : float
    height_filter_min : float
    height_filter_max : float
    downsample_rate : int
    sor_radius : float
    sor_min_neighbors : int
    intensity_min : float
    """
    max_range: float
    min_range: float
    height_filter_min: float
    height_filter_max: float
    downsample_rate: int
    sor_radius: float
    sor_min_neighbors: int
    intensity_min: float


def apply_filter(points: numpy.ndarray, config: FilterConfig) -> numpy.ndarray:
    """
    Apply filtering to a point cloud.

    Parameters
    ----------
    points : np.ndarray
        2D array of shape (N, D) where D >= 3, dtype=float64.
        The first three columns are x, y, z coordinates; optional fourth column can be intensity.
    config : FilterConfig
        Filtering configuration.

    Returns
    -------
    np.ndarray
        2D array of shape (M, D) containing filtered points, dtype=float64.
        Memory is allocated internally and managed via a PyCapsule.
    """
    ...