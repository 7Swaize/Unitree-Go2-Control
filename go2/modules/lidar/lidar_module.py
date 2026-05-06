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
    def __init__(self) -> None:
        super().__init__("LIDAR")
        self._ros_proc = None
        self._dispatcher = None
        self._iox_receiver = None

    @override
    def _initialize(self) -> None:
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
        self._dispatcher._register_decoded(callback)

    def register_filtered_pointcloud_callback(self, callback: Callable[[int, np.ndarray], None]) -> None:
        self._dispatcher._register_filtered(callback)

    def register_synced_pointcloud_callback(self, callback: Callable[[int, np.ndarray, np.ndarray], None]) -> None:
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

