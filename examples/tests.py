import time
import cv2
import numpy as np

from go2.core import Go2Controller, ModuleType
from go2.modules.video import CameraSourceFactory, StreamConfig



class Tests:
    def __init__(self):
        self.unitree_controller = Go2Controller(use_sdk=False)
        self.unitree_controller.register_cleanup_callback(self.shutdown_callback)

        self.unitree_controller.add_module(ModuleType.AUDIO)
        self.unitree_controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_camera_group({
            "depth": CameraSourceFactory.create_depth_camera(),
            "opencv": CameraSourceFactory.create_opencv_camera()
        }))
        # self.unitree_controller.add_module(ModuleType.LIDAR, use_sdk=True)
        
        self.unitree_controller.video.start_stream_server()
        self.unitree_controller.video.get_stream_server_local_ip()
        print(f"WebRTC streaming at: http://{self.unitree_controller.video.get_stream_server_local_ip()}:{self.unitree_controller.video.get_stream_server_port()}")
        

    def test_depth_camera(self):
        try:
            while True:
                frame_result = self.unitree_controller.video.get_frames()
                print(frame_result)

                if frame_result.has_any:
                    color, depth = frame_result.color, frame_result.depth

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
            frame_result = self.unitree_controller.video.get_frames()["opencv"]
            if not frame_result.has_color:
                continue
            
            self.unitree_controller.video.send_frame(frame_result.color)

    
    def test_audio(self):
        while True:
            self.unitree_controller.audio.play_audio("Hello World")
            time.sleep(1)


    def shutdown_callback(self):
        cv2.destroyAllWindows()
        time.sleep(1)


if __name__ == '__main__':
    tests = Tests()
    tests.test_streaming()
