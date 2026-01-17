import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.core.unitree_control_core import UnitreeGo2Controller
from src.core.module_registry import ModuleType
from src.video_control.camera_source_factory import CameraSourceFactory

import time
import cv2
import numpy as np


class Tests:
    def __init__(self):
        self.unitree_controller = UnitreeGo2Controller(use_sdk=False)
        self.unitree_controller.register_cleanup_callback(self.shutdown_callback)

        self.unitree_controller.add_module(ModuleType.AUDIO)
        self.unitree_controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_depth_camera())
        
        self.unitree_controller.video.start_stream_server()
        self.unitree_controller.video.get_stream_server_local_ip()
        print(f"WebRTC streaming at: http://{self.unitree_controller.video.get_stream_server_local_ip()}:{self.unitree_controller.video.get_stream_server_port()}")
        

    def test_depth_camera(self):
        try:
            while True:
                status, frames = self.unitree_controller.video.get_frames()

                if status == 0 and frames is not None:
                    color, depth = frames

                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth, alpha=0.03),
                        cv2.COLORMAP_JET
                    )

                    combined = np.hstack((
                        color,
                        cv2.resize(depth_colormap, (color.shape[1], color.shape[0]))
                    ))

                    cv2.imshow("RealSense Depth Camera", combined)
                    self.unitree_controller.video.send_frame(combined)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()

    
    def test_streaming(self):
        while True:
            status, image = self.unitree_controller.video.get_frames()
            if status != 0 or image is None:
                continue
            
            self.unitree_controller.video.send_frame(image)

    
    def test_audio(self):
        while True:
            self.unitree_controller.audio.play_audio("Hello World")
            time.sleep(1)


    def shutdown_callback(self):
        cv2.destroyAllWindows()
        time.sleep(1)


if __name__ == '__main__':
    tests = Tests()
    tests.test_depth_camera()
