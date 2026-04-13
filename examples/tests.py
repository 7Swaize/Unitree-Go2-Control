import time
import cv2
import numpy as np

from go2.core import Go2Controller, ModuleType, HardwareType
from go2.modules.video import CameraSourceFactory



class Tests:
    def __init__(self):
        self.controller = Go2Controller(hardware_type=HardwareType.VIRTUAL)
        self.controller.register_cleanup_callback(self.shutdown_callback)

        self.controller.add_module(ModuleType.AUDIO)
        self.controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_camera_group({
            "sim": CameraSourceFactory.create_depth_camera()
        }))
        
        self.controller.video.start_stream_server()
        self.controller.video.get_stream_server_local_ip()
        print(f"WebRTC streaming at: http://{self.controller.video.get_stream_server_local_ip()}:{self.controller.video.get_stream_server_port()}")
        

    def test_depth_camera(self):
        try:
            while True:
                frame_result = self.controller.video.get_frames()["depth"]

                if frame_result.is_fully_valid():
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
                    self.controller.video.send_frame(combined)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()

    
    def test_streaming(self):
        try:
            while True:
                frame_result = self.controller.video.get_frames()["sim"]

                if not frame_result.has_color():
                    continue
                
                cv2.imshow("RealSense Depth Camera", frame_result.color)
                self.controller.video.send_frame(frame_result.color)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        finally:
            self.controller.safe_shutdown()


    def test_audio(self):
        while True:
            self.controller.audio.play_audio("Hello World")
            time.sleep(1)

    def test_movement(self):
        time.sleep(3)
        self.controller.movement.stand_up()
        time.sleep(5)
        self.controller.movement.stand_down()

    def shutdown_callback(self):
        cv2.destroyAllWindows()
        time.sleep(1)


if __name__ == '__main__':
    tests = Tests()
    tests.test_streaming()
