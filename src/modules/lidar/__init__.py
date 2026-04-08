"""LIDAR Module for Student Use"""

import os
import signal
import subprocess
import numpy as np
from typing import Callable, Any

from core.module import DogModule


class LIDARModule(DogModule):
    def __init__(self, use_sdk: bool = True):
        super().__init__("LIDAR")
        self.use_sdk = use_sdk
        self.ros_proc = None
        self.dispatcher = None
        self.zmq_receiver = None


    def initialize(self):
        if self._initialized:
            return
        
        if not self.use_sdk:
            raise RuntimeError("[LIDAR] Cannot utilize LIDAR module without Unitree SDK")
    
        self.launch_ros2_internal()
        # Note: Full lidar implementation would be added here with callback dispatcher
        self._initialized = True

    
    def launch_ros2_internal(self) -> None:
        self.ros_proc = subprocess.Popen(
            ["ros2", "launch", "bringup", "lidar_processor.launch.py"],
            preexec_fn=os.setsid
        )


    def register_decoded_pointcloud_callback(self, callback: Callable[[int, np.ndarray], Any]) -> None:
        if self.dispatcher:
            self.dispatcher.register_decoded(callback)

    
    def register_filtered_pointcloud_callback(self, callback: Callable[[int, np.ndarray], Any]) -> None:
        if self.dispatcher:
            self.dispatcher.register_filtered(callback)
        

    def shutdown(self):
        if self.ros_proc and self.ros_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.ros_proc.pid), signal.SIGINT)
                self.ros_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.ros_proc.pid), signal.SIGTERM)
                self.ros_proc.wait(timeout=5)  
        
        self.ros_proc = None
        self._initialized = False


__all__ = ["LIDARModule"]
