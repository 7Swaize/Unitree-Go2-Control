import sys
import cv2
import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


class Main():
    def __init__(self):
        self.lower_blue = np.array([100, 150, 0]) # i just took these from google
        self.upper_blue = np.array([140, 255, 255])

        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()

        self.sport_client = SportClient()
        self.sport_client.SetTimeout(5.0)
        self.sport_client.Init()


    def main(self):
        print("Press 'q' to quit.")

        while True:
            code, data = self.video_client.GetImageSample()
            if data is None or code != 0:
                continue

            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                # only moves if the detected blob is large enough (filters any background noise)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(largest)
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)

                    # make robot move in direction of blob
                    self.move_towards_blob(cx, cy, w, h, frame.shape[1])

                # draw rectangle around largest blob
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow('Largest Colored Blob', frame)

            if (cv2.waitKey(1)) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.sport_client.Move(0, 0, 0)


    def move_towards_blob(self, cx, cy, w, h, frame_width):
        center_tolerance = frame_width * 0.2
        desired_area = 10000

        center_x = frame_width / 2
        offset_x = cx - center_x

        if w * h > desired_area:
            return
        
        if offset_x < -center_tolerance:
            self.sport_client.Move(0.0, 0.0, 0.3)  # rotate left

        elif offset_x > center_tolerance:
            self.sport_client.Move(0.0, 0.0, -0.3)  # rotate right

        else:
            print("Moving forward...")
            self.sport_client.Move(0.2, 0.0, 0.0)  # forward motion

        time.sleep(0.1)
        # self.sport_client.Move(0, 0, 0) maybe uncomment this for smoother movement???


if __name__ == "__main__":
    main = Main()
    main.main()