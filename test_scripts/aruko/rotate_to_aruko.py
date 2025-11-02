import time
import sys
import struct
import math
import threading
import cv2
import numpy as np

from dataclasses import dataclass
from enum import Enum



aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


# define all the mapping in here
class MarkerMappings(Enum):
    STOP_MARKER = 0


class DogFunctionalityWrapper:
    def __init__(self):
        global LowState_, ChannelSubscriber
        self.stop_event = threading.Event()
        self.use_unitree_sdk_methods = False 

        # If sdk imports are unable to be executed, it falls back to using cv2 and a standard webcam.
        # So, if I execute this script while on a remote desktop connection with the dog it will execute with movement commands.
        # However, if I run it on any machine without the sdk installed, it will execute with cv2. 
        try:
            # lazy load for now
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
            from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            from unitree_sdk2py.go2.video.video_client import VideoClient

            LowState_ = LowState_
            ChannelSubscriber = ChannelSubscriber

            print("[Init] Unitree SDK detected. Using real dog control.")
            self.use_unitree_sdk_methods = True

            print("[Init] Initializing ChannelFactory")
            if len(sys.argv) > 1:
                ChannelFactoryInitialize(0, sys.argv[1])
            else:
                ChannelFactoryInitialize(0)

            print("[Init] Creating SportClient")
            self.sport_client = SportClient()
            self.sport_client.Init()
            self.sport_client.SetTimeout(3.0)

            print("[Init] Connecting stop key override")
            self.handler = UnitreeRemoteController.CustomHandler(self.stop_event)
            self.handler.init()

            print("[Init] Standing up and stopping movement")
            self.sport_client.StandUp()
            time.sleep(1)
            self.sport_client.StopMove()
            time.sleep(0.5)

            print("[Init] Initializing VideoClient")
            self.video_client = VideoClient()
            self.video_client.SetTimeout(3.0)
            self.video_client.Init()

        except ImportError:
            print("[Init] Unitree SDK not found. Running in simulated (webcam) mode.")

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open webcam")


    def get_image(self):
        if self.use_unitree_sdk_methods:
            code, data = self.video_client.GetImageSample()
            if code != 0 or data is None:
                return -1, None

            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            return code, image

        ret, image = self.cap.read()
        if not ret:
            return -1, None
        
        return 0, image

    def rotate_dog(self, amount):
        # print(f"rotating {amount}")

        if self.use_unitree_sdk_methods:
            vx, vy, vz = 0.0, 0.0, amount
            self.sport_client.Move(vx, vy, vz)
            

    def shift_dog(self, amount_x=0, amount_y=0):
        # print(f"Shifting: {amount_x}, {amount_y}")

        if self.use_unitree_sdk_methods:
            vx, vy, vz = amount_x, amount_y, 0.0
            self.sport_client.Move(vx, vy, vz)


    def stop_dog(self):
        print("Stopping")

        if self.use_unitree_sdk_methods:
            self.sport_client.StopMove()

    def cleanup(self):
        print("[Cleanup] Cleaning up resources.")

        self.stop_event.clear()
        if not self.use_unitree_sdk_methods:
            self.cap.release()


class UnitreeRemoteController:
        def __init__(self):
            # key
            self.Lx = 0           
            self.Rx = 0            
            self.Ry = 0            
            self.Ly = 0

            # button
            self.L1 = 0
            self.L2 = 0
            self.R1 = 0
            self.R2 = 0
            self.A = 0
            self.B = 0
            self.X = 0
            self.Y = 0
            self.Up = 0
            self.Down = 0
            self.Left = 0
            self.Right = 0
            self.Select = 0
            self.F1 = 0
            self.F3 = 0
            self.Start = 0

        def parse_botton(self, data1, data2):
            # uint8_t bit seq, each flag representing an input.
            self.R1 = (data1 >> 0) & 1
            self.L1 = (data1 >> 1) & 1
            self.Start = (data1 >> 2) & 1
            self.Select = (data1 >> 3) & 1
            self.R2 = (data1 >> 4) & 1
            self.L2 = (data1 >> 5) & 1
            self.F1 = (data1 >> 6) & 1
            self.F3 = (data1 >> 7) & 1

            self.A = (data2 >> 0) & 1
            self.B = (data2 >> 1) & 1
            self.X = (data2 >> 2) & 1
            self.Y = (data2 >> 3) & 1
            self.Up = (data2 >> 4) & 1
            self.Right = (data2 >> 5) & 1
            self.Down = (data2 >> 6) & 1
            self.Left = (data2 >> 7) & 1

        def parse_key(self,data):
            lx_offset = 4
            self.Lx = struct.unpack('<f', data[lx_offset:lx_offset + 4])[0]
            rx_offset = 8
            self.Rx = struct.unpack('<f', data[rx_offset:rx_offset + 4])[0]
            ry_offset = 12
            self.Ry = struct.unpack('<f', data[ry_offset:ry_offset + 4])[0]
            L2_offset = 16
            L2 = struct.unpack('<f', data[L2_offset:L2_offset + 4])[0] # Placeholder，unused
            ly_offset = 20
            self.Ly = struct.unpack('<f', data[ly_offset:ly_offset + 4])[0]

        def parse(self,remoteData):
            self.parse_key(remoteData)
            self.parse_botton(remoteData[2], remoteData[3])

        class CustomHandler:
            def __init__(self, stop_event: threading.Event):
                self.remote_controller = UnitreeRemoteController()
                self.stop_event = stop_event

            def init(self):
                self.lowstate_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
                self.lowstate_subscriber.Init(self.lowstate_callback, 10)
            
            def lowstate_callback(self, msg): # msg: LowState_
                self.remote_controller.parse(msg.wireless_remote)

                if self.remote_controller.A == 1:
                    self.stop_event.set()


def extract_corners_points(corners):
    x0, y0 = corners[0]
    x1, y1 = corners[1]
    x2, y2 = corners[2]
    x3, y3 = corners[3]

    return (x0, y0), (x1, y1), (x2, y2), (x3, y3)


def determine_horizontal_offset(image, fiducial_center, screen_width):
    midX = screen_width // 2
    cx = int(fiducial_center[0])
    return midX - cx


def compute_fiducial_center_point(p0, p1):
    mx = int((p0[0] + p1[0]) / 2)
    my = int((p0[1] + p1[1]) / 2)
    return (mx, my)


def dist_between_points(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]

    return math.sqrt(dx*dx + dy*dy)


def get_fiducial_area_from_corners(c0, c1, c2, c3):
    width = dist_between_points(c0, c1)
    height = dist_between_points(c1, c2)
    
    return width * height


def get_aruko_marker(image):
    if len(image.shape) == 2: 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    return corners, ids, rejected


def extract_data_from_marker(image, corners, ids):
    if corners is None or len(corners) == 0 or ids is None or len(ids) == 0:
        return -1, [(0, 0)], (0, 0), 0, 0

    bounds = [(int(x), int(y)) for x, y in corners[0][0]]
    marker_id = int(ids[0][0])

    c0, c1, c2, c3 = extract_corners_points(corners[0][0])
    fiducial_center = compute_fiducial_center_point(c0, c2)
    h_offset = determine_horizontal_offset(image, fiducial_center, image.shape[1])
    fiducial_area = get_fiducial_area_from_corners(c0, c1, c2, c3)

    return marker_id, bounds, fiducial_center, h_offset, fiducial_area


def scan_for_aruko_marker(functionality_wrapper: DogFunctionalityWrapper,
                          window_title,
                          search_range,
                          search_delta,
                          max_sweeps=3):
    print("Scanning for Marker")

    fiducial_found = False
    marker_id = -1

    current_angle = 0  
    sweeps_done = 0
    direction = 1  

    # first sweep: half of search_range in initial direction
    half_range = search_range / 2
    sweep_limit = half_range

    while not fiducial_found and sweeps_done < max_sweeps:
        if functionality_wrapper.stop_event.is_set():
            break

        code, image = functionality_wrapper.get_image()
        if code != 0 or image is None:
            continue

        corners, ids, _ = get_aruko_marker(image)
        marker_id, bounds, _, _, _ = extract_data_from_marker(image, corners, ids)
        fiducial_found = (marker_id != -1)

        if fiducial_found:
            cv2.putText(image, f"ID: {marker_id}", bounds[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow(window_title, image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        functionality_wrapper.rotate_dog(direction * search_delta)
        current_angle += direction * search_delta
        next_angle = current_angle + direction * search_delta

        # checks next step would exceed the sweep limit
        if abs(next_angle) >= sweep_limit:
            next_angle = direction * sweep_limit
            sweeps_done += 1
            direction *= -1  # reverse sweep direction

        time.sleep(0.1)

    return fiducial_found, marker_id



def walk_to_aruko_marker(functionality_wrapper: DogFunctionalityWrapper, window_title):
    print("Moving to Marker")

    arrived = False
    marker_id = -1

    forward_step = 1
    rotate_step = 3
    h_offset_threshold = 30 # i just played around with this number; in the future we would need a system to calculate this
    fiducial_area_threshold = 60000 # i just played around with this number; in the future we would need a system to calculate this

    while not arrived:
        if functionality_wrapper.stop_event.is_set():
            break

        code, image = functionality_wrapper.get_image()
        if code != 0 or image is None:
            continue

        corners, ids, _ = get_aruko_marker(image)
        marker_id, bounds, fiducial_center, h_offset, fiducial_area = extract_data_from_marker(image, corners, ids)

        if marker_id == -1:
            break  # no marker detected

        if h_offset < -h_offset_threshold:
            functionality_wrapper.rotate_dog(rotate_step)
        elif h_offset > h_offset_threshold:
            functionality_wrapper.rotate_dog(-rotate_step)

        if fiducial_area < fiducial_area_threshold:
            functionality_wrapper.shift_dog(forward_step)
        else:
            arrived = True

        cv2.putText(image, f"ID: {marker_id}", (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow(window_title, image)

        time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    return arrived, marker_id


def respond_to_aruko_id(functionality_wrapper: DogFunctionalityWrapper, marker_id):
    print("Responding to Aruko Marker")
    
    if (marker_id == MarkerMappings.STOP_MARKER.value):
        functionality_wrapper.stop_dog() 
        time.sleep(2)
        return True
        

    time.sleep(2)
    return False


## MOVE
def mainMove():
    input("Press Enter to continue...")

    functionality_wrapper = DogFunctionalityWrapper()
    window_title = "Aruko Detection"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    search_range = 70
    search_delta = 1
    has_arrived = False

    while True:
        fiducial_found, marker_id = scan_for_aruko_marker(functionality_wrapper, window_title, search_range, search_delta, max_sweeps=3) # scans for 3 sweeps. if it doesnt find anything it quits

        if fiducial_found:
            has_arrived, marker_id = walk_to_aruko_marker(functionality_wrapper, window_title)
            print(marker_id)

            if has_arrived:
                should_exit = respond_to_aruko_id(functionality_wrapper, marker_id)
                if should_exit:
                    break
        else:
            break

    cv2.destroyAllWindows()
   
    functionality_wrapper.stop_dog()
    time.sleep(0.5)

    functionality_wrapper.cleanup()



if __name__ == "__main__":
    mainMove()

