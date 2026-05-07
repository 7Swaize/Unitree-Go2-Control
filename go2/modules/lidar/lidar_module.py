import os
import sys
import signal
import subprocess
import numpy as np
from typing import Callable, Any
from typing_extensions import override

from ...core.module import DogModule
from .callback_dispatcher import CallbackDispatcher
from .iox_receiver import IoxReceiver


class LIDARModule(DogModule):
    """
    ``LIDARModule`` provides a simple API for:
        - Recieving decoded `PointCloud2` structures as xyz-intensity numpy arrays
        - Recieving filtered `PointCloud2` structures as xyz-intensity numpy arrays

    Users should interact **only** with this class and should not directly use other means of accessing lidar.
    
    Users should not access or construct this class directly.
    Rather, they should access it through the :class:`~core.controller.Go2Controller` instance.

    Notes
    -----
    - This module launches ROS2 nodes.
      It is **critical** that students **ALWAYS** call :meth:`Go2Controller.safe_shutdown` after normal (error free) script exit.
    """

    def __init__(self) -> None:
        super().__init__("LIDAR")
        self._ros_proc = None
        self._dispatcher = None
        self._iox_receiver = None

    @override
    def _initialize(self) -> None:
        """
        Initialize the lidar module. This is called internally,
        and should not be called directly by users.

        This method starts ROS2 process and Iceoryx2 pub-sub nodes for lidar data transfer.
        """
        if self._initialized:
            return

        self._launch_ros()
        self._launch_bridge()

        self._initialized = True

    def _launch_ros(self) -> None:
        kwargs = dict()
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # To detach child process: https://stackoverflow.com/questions/45911705/why-use-os-setsid-in-python
            kwargs["start_new_session"] = True 

        self._ros_proc = subprocess.Popen(
            ["ros2", "launch", "bringup", "lidar_processor.launch.py"],
            **kwargs
        )

    def _launch_bridge(self) -> None:
        self._dispatcher = CallbackDispatcher()
        self._iox_receiver = IoxReceiver(self._dispatcher)

        self._iox_receiver.start()


    def register_decoded_pointcloud_callback(self, callback: Callable[[int, np.ndarray], None]) -> None:
        """
        Register a callback to receive decoded PointCloud2 data.

        The callback is triggered whenever a new raw point cloud sample is received
        via Iceoryx2 and successfully reshaped.

        Parameters
        ----------
        callback : Callable[[int, np.ndarray], None]
            A function to be called with:
                - **timestamp** (int): The source timestamp in nanoseconds.
                - **points** (np.ndarray): A float64 array of shape ``(N, 3)`` 
                  for [x, y, z] or ``(N, 4)`` if intensity is supported 
                  [x, y, z, intensity].
        """
        self._dispatcher._register_decoded(callback)

    def register_filtered_pointcloud_callback(self, callback: Callable[[int, np.ndarray], None]) -> None:
        """
        Register a callback to receive filtered PointCloud2 data.

        This receives data after it has passed through a Statistical Outlier Removal (SOR) filter.
        The filter calculates the mean distance to the nearest neighbors and trims points that fall outside a specific global 
        percentage or standard deviation threshold.

        Parameters
        ----------
        callback : Callable[[int, np.ndarray], None]
            A function to be called with:
                - **timestamp** (int): The source timestamp in nanoseconds.
                - **points** (np.ndarray): A float64 array of shape ``(N, 3)`` 
                  or ``(N, 4)`` containing the points that passed the 
                  outlier filter.
        """
        self._dispatcher._register_filtered(callback)

    def register_synced_pointcloud_callback(self, callback: Callable[[int, np.ndarray, np.ndarray], None]) -> None:
        """
        Register a callback to receive synchronized raw and SOR-filtered point clouds.

        This is particularly useful for debugging or visualization, allowing a direct 
        comparison between the raw sensor data and the data remaining after the 
        outlier removal percentage is applied.

        Parameters
        ----------
        callback : Callable[[int, np.ndarray, np.ndarray], None]
            A function to be called with:
                - **timestamp** (int): The common timestamp in nanoseconds.
                - **decoded_points** (np.ndarray): The raw float64 array of shape 
                ``(N, 3)`` or ``(N, 4)``.
                - **filtered_points** (np.ndarray): The SOR-filtered float64 array of 
                shape ``(N, 3)`` or ``(N, 4)``.
        """
        self._dispatcher._register_synced(callback)


    @override
    def _shutdown(self) -> None:
        if self._ros_proc and self._ros_proc.poll() is None:
            try:
                if sys.platform == "win32":
                    self._ros_proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self._ros_proc.pid), signal.SIGINT)
                self._ros_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._ros_proc.terminate()
                self._ros_proc.wait(timeout=5)

        if self._iox_receiver:
            self._iox_receiver._shutdown()
            self._iox_receiver.join(timeout=2)

        self._initialized = False

