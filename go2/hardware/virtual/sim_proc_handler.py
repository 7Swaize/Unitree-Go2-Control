import os
import signal
import subprocess
from typing import Optional


class SimProcHandler:
    def __init__(self) -> None:
        self._sim_proc: Optional[subprocess.Popen] = None

    def _start_sim(self) -> None:
        self._sim_proc = subprocess.Popen(
            ["/home/gsmst/unitree_mujoco/simulate/build/unitree_mujoco", "-r", "go2", "-s", "scene_terrain.xml"], # TODO: Refactor into using git submodule and scikit build
        )


    def _shutdown(self) -> None:
        if self._sim_proc and self._sim_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._sim_proc.pid), signal.SIGINT)
                self._sim_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._sim_proc.pid), signal.SIGTERM)
                self._sim_proc.wait(timeout=5)
        
        self._sim_proc = None