import os
import sys
import time
from enum import Enum

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from aruko_helpers import *

from unitree_control.core.unitree_control_core import UnitreeGo2Controller
from unitree_control.core.module_registry import ModuleType
from unitree_control.states.dog_state_abstract import DogStateAbstract
from unitree_control.video_control.video_module import VideoModule


class MarkerMappings(Enum):
    STOP = 0
    RIGHT_90_DEGREES = 1
    LEFT_90_DEGREES = 2
    ROTATE_180_DEGREES = 3
    MARKER_5 = 4


class ScanForMarkerState(DogStateAbstract):
    def __init__(self, functionality_wrapper, window_title, search_range, search_delta, max_sweeps=3):
        super().__init__(functionality_wrapper)

        self.window_title = window_title
        self.search_range = search_range
        self.search_delta = search_delta
        self.max_sweeps = max_sweeps
        self.target_marker_id = None


    def set_target_marker_id(self, marker_id):
        self.target_marker_id = marker_id


    def execute(self):
        if self.target_marker_id is None:
            print("[ScanForMarkerState] Target marker ID not set.")
            return

        fiducial_found = False
        marker_id = -1
        current_angle = 0  
        sweeps_done = 0

        while not fiducial_found and sweeps_done < self.max_sweeps:
            status, frames = self.unitree_controller.video.get_frames()
            if status != 0 or frames is None:
                continue

            color_image, depth_image = frames
            