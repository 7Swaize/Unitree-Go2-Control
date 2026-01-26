import os
import signal
import subprocess

from src.core.base_module import DogModule

# pip install pyzmq

class LIDARModule(DogModule):
    def __init__(self, use_sdk: bool = True):
        super().__init__("LIDAR")
        self.use_sdk = use_sdk
        self._ros_proc = None


    def initialize(self):
        if self._initialized:
            return
        
        if not self.use_sdk:
            raise RuntimeError("[LIDAR] Cannot utilize LIDAR module without Unitree SDK")
    
        self._launch_ros2_internal()
        self._initialized = True


    def _launch_ros2_internal(self) -> None:
        self._ros_proc = subprocess.Popen(
            ["ros2", "launch", "bringup", "lidar_processor.launch.py"],
            preexec_fn=os.setsid
        )
        

    def shutdown(self):
        if not self._ros_proc:
            return

        if self._ros_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._ros_proc.pid), signal.SIGINT)
                self._ros_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._ros_proc.pid), signal.SIGTERM)
                self._ros_proc.wait(timeout=5)  
        
        self._ros_proc = None
        self._initialized = False