import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

from test_scripts.callback_manager import InputHandler, InputSignal
from test_scripts.input_handle import ControllerState


a_pressed: bool = False

def main():
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    sport_client = SportClient()
    sport_client.Init()

    # TODO: make callback handle variadic parameters
    input_handler = InputHandler()
    input_handler.register_callback(
        InputSignal.BUTTON_A,
        on_button_A,
        name="on_button_",
    )

    vx, vy, vz = 0.3, 0.0, 0.0
    start_time = time.time()

    while time.time() - start_time < 3.0:
        if a_pressed:
            break

        sport_client.Move(vx, vy, vz)
        time.sleep(0.05) 

    input_handler.shutdown()

def on_button_A(state: ControllerState):
    global a_pressed
    a_pressed = True
    
if __name__ == "__main__":
    main()