Module Configuration & Parameters
==================================

When adding optional modules to your controller, you often need to pass constructor parameters that configure the module's behavior. This guide explains what each module requires and how to configure them.

Overview
--------

Each module has specific initialization requirements:

- **Simple modules** (Audio, OCR) - No parameters required
- **Configurable modules** (Video) - Require configuration objects
- **Hardware modules** (Input) - May depend on SDK mode

Quick Reference
---------------

===================== ================= ===============================================
Module                Parameters        Notes
===================== ================= ===============================================
**Movement**          None              Always available by default
**Input Control**     None              Available by default in SDK mode
**Audio**             None              No configuration needed
**OCR**               None              No configuration needed
**Video**             camera_source     Requires a CameraSource instance
===================== ================= ===============================================

Detailed Module Configuration
------------------------------

Audio Module
^^^^^^^^^^^^

The audio module requires no parameters and is ready to use immediately:

.. code-block:: python

    from src.core.unitree_control_core import UnitreeGo2Controller
    from src.core.module_registry import ModuleType

    controller = UnitreeGo2Controller(use_sdk=True)
    controller.add_module(ModuleType.AUDIO)
    
    controller.audio.play_audio("Hello World")
    controller.safe_shutdown()

OCR Module
^^^^^^^^^^

The OCR (Optical Character Recognition) module requires no parameters:

.. code-block:: python

    from src.core.unitree_control_core import UnitreeGo2Controller
    from src.core.module_registry import ModuleType
    import numpy as np

    controller = UnitreeGo2Controller(use_sdk=True)
    controller.add_module(ModuleType.OCR)
    
    # Assuming you have a video frame
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    text, highlighted_image = controller.ocr.extract_text_from_image(image)
    
    controller.safe_shutdown()

Video Module (With Camera Configuration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The video module requires a ``camera_source`` parameter. Use the ``CameraSourceFactory`` to create appropriate camera sources:

.. code-block:: python

    from src.core.unitree_control_core import UnitreeGo2Controller
    from src.core.module_registry import ModuleType
    from src.video_control.camera_source_factory import CameraSourceFactory

    controller = UnitreeGo2Controller(use_sdk=True)
    
    # Create a camera source from the robot's SDK camera
    camera = CameraSourceFactory.create_sdk_camera()
    controller.add_module(ModuleType.VIDEO, camera_source=camera)
    
    # Get frames
    status, frames = controller.video.get_frames()
    if status == 0:
        print(f"Frame shape: {frames.shape}")
    
    controller.safe_shutdown()

**Available Camera Sources:**

Use the ``CameraSourceFactory`` to create different camera types:

.. code-block:: python

    from src.video_control.camera_source_factory import CameraSourceFactory

    # Robot's internal SDK camera (most common on robot hardware)
    sdk_camera = CameraSourceFactory.create_sdk_camera()
    
    # OpenCV camera (USB webcam, laptop camera, etc.)
    # camera_index defaults to 0 (first camera)
    opencv_camera = CameraSourceFactory.create_opencv_camera(camera_index=0)
    
    # Intel RealSense depth camera (returns both color and depth frames)
    depth_camera = CameraSourceFactory.create_depth_camera()

Then pass the camera source to the module:

.. code-block:: python

    # With SDK camera
    controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_sdk_camera())
    
    # With webcam
    controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_opencv_camera())
    
    # With depth camera
    controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_depth_camera())

**Getting Frames:**

After adding the video module, retrieve frames using ``get_frames()``:

.. code-block:: python

    # For single-frame cameras (SDK or OpenCV)
    status, frame = controller.video.get_frames()
    if status == 0 and frame is not None:
        print(f"Image shape: {frame.shape}")  # (height, width, 3)
    
    # For depth cameras (returns color + depth)
    status, frames = controller.video.get_frames()
    if status == 0 and isinstance(frames, tuple):
        color, depth = frames
        print(f"Color: {color.shape}, Depth: {depth.shape}")

**Streaming Video:**

To stream video to a browser via WebRTC:

.. code-block:: python

    controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_sdk_camera())
    
    # Start the streaming server
    controller.video.start_stream_server()
    
    # Get server info
    ip = controller.video.get_stream_server_local_ip()
    port = controller.video.get_stream_server_port()
    print(f"Stream available at: http://{ip}:{port}")
    
    try:
        while True:
            status, frame = controller.video.get_frames()
            if status == 0 and frame is not None:
                # Send frame to streaming clients
                controller.video.send_frame(frame)
            
            time.sleep(0.033)  # ~30 FPS
    finally:
        controller.safe_shutdown()

Input Control Module
^^^^^^^^^^^^^^^^^^^^

The input control module handles gamepad/joystick input. It's available by default in SDK mode and requires no additional parameters:

.. code-block:: python

    from src.core.unitree_control_core import UnitreeGo2Controller
    from src.controller_input_control.input_signal import InputSignal

    controller = UnitreeGo2Controller(use_sdk=True)
    
    # Input module is already loaded
    
    def on_button_x(state):
        """Called when button X is pressed."""
        print(f"Button X pressed: {state.x}")
    
    def on_left_stick_moved(state):
        """Called when left stick moves."""
        print(f"Left stick: X={state.lx}, Y={state.ly}")
    
    # Register callbacks for specific inputs
    controller.input.register_callback(InputSignal.BUTTON_X, on_button_x)
    controller.input.register_callback(InputSignal.LEFT_STICK, on_left_stick_moved)
    
    # Keep program running to process callbacks
    try:
        while True:
            time.sleep(0.1)
    finally:
        controller.safe_shutdown()

**Available Input Signals:**

The ``InputSignal`` enum provides all available controller inputs:

.. code-block:: python

    from src.controller_input_control.input_signal import InputSignal
    
    # Button signals
    InputSignal.BUTTON_A
    InputSignal.BUTTON_B
    InputSignal.BUTTON_X
    InputSignal.BUTTON_Y
    
    # Stick signals
    InputSignal.LEFT_STICK
    InputSignal.RIGHT_STICK
    InputSignal.LEFT_STICK_X
    InputSignal.LEFT_STICK_Y
    InputSignal.RIGHT_STICK_X
    InputSignal.RIGHT_STICK_Y
    
    # Triggers and bumpers
    InputSignal.LEFT_TRIGGER
    InputSignal.RIGHT_TRIGGER
    InputSignal.LEFT_BUMPER
    InputSignal.RIGHT_BUMPER
    
    # D-pad
    InputSignal.DPAD_UP
    InputSignal.DPAD_DOWN
    InputSignal.DPAD_LEFT
    InputSignal.DPAD_RIGHT
    
    # Other buttons
    InputSignal.SELECT
    InputSignal.START
    InputSignal.F1
    InputSignal.F3

**Controller State Information:**

The callback receives a ``ControllerState`` object with current values:

.. code-block:: python

    from src.controller_input_control.input_signal import InputSignal
    
    def on_input(state):
        # Access stick positions (-1.0 to 1.0)
        print(f"Left stick: lx={state.lx}, ly={state.ly}")
        print(f"Right stick: rx={state.rx}, ry={state.ry}")
        
        # Access button states (0.0 or 1.0)
        print(f"Button A: {state.a}")
        print(f"D-pad up: {state.up}")
        
        # Access triggers (0.0 to 1.0)
        print(f"Left trigger: {state.l2}")
        print(f"Right trigger: {state.r2}")
    
    controller.input.register_callback(InputSignal.BUTTON_A, on_input)

ArUco Module
^^^^^^^^^^^^

The ArUco marker detection module works with video frames and requires no special parameters:

.. code-block:: python

    from src.core.unitree_control_core import UnitreeGo2Controller
    from src.core.module_registry import ModuleType
    from src.video_control.camera_source_factory import CameraSourceFactory

    controller = UnitreeGo2Controller(use_sdk=True)
    
    # Add video module with a camera
    camera = CameraSourceFactory.create_sdk_camera()
    controller.add_module(ModuleType.VIDEO, camera_source=camera)
    
    # Get frames and detect markers
    status, frames = controller.video.get_frames()
    if status == 0:
        color_frame = frames
        # Use ArUco helper functions with the frame
        from src.aruko.aruko_helpers import get_aruko_marker
        
        corners, ids, rejected = get_aruko_marker(color_frame)

Complete Example: Multi-Module Setup
-------------------------------------

This example shows how to add multiple modules with their required parameters:

.. code-block:: python

    import time
    from src.core.unitree_control_core import UnitreeGo2Controller
    from src.core.module_registry import ModuleType
    from src.video_control.camera_source_factory import CameraSourceFactory
    from src.controller_input_control.input_signal import InputSignal

    # Initialize controller
    controller = UnitreeGo2Controller(use_sdk=True)
    
    # Add optional modules with their parameters
    try:
        # Video module requires a camera source
        camera = CameraSourceFactory.create_sdk_camera()
        controller.add_module(ModuleType.VIDEO, camera_source=camera)
        
        # Audio module requires no parameters
        controller.add_module(ModuleType.AUDIO)
        
        # OCR module requires no parameters
        controller.add_module(ModuleType.OCR)
        
        # All modules are now ready
        controller.movement.stand_up()
        time.sleep(1)
        
        # Use modules
        status, frame = controller.video.get_frames()
        if status == 0:
            controller.audio.play_audio("I can see you!")
        
        # Register input callback
        def on_button_x(state):
            text, image = controller.ocr.extract_text_from_image(frame)
            print(f"Detected text: {text}")
        
        controller.input.register_callback(InputSignal.BUTTON_X, on_button_x)
        
        # Keep running
        while True:
            time.sleep(0.1)
            
    finally:
        controller.safe_shutdown()
        
    finally:
        controller.safe_shutdown()

Common Patterns for Module Parameters
--------------------------------------

Parameter Passing with ``add_module()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters are passed as keyword arguments to ``add_module()``:

.. code-block:: python

    # Single parameter
    controller.add_module(ModuleType.VIDEO, camera_source=my_camera)
    
    # The parameter name must match the module's ``__init__`` parameter

Creating Reusable Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For complex setups, create helper functions:

.. code-block:: python

    from src.core.module_registry import ModuleType
    from src.video_control.camera_source_factory import CameraSourceFactory
    
    def setup_video_module(controller, camera_type='sdk'):
        """Helper to set up video with specified camera type."""
        if camera_type == 'sdk':
            camera = CameraSourceFactory.create_sdk_camera()
        elif camera_type == 'opencv':
            camera = CameraSourceFactory.create_opencv_camera()
        elif camera_type == 'depth':
            camera = CameraSourceFactory.create_depth_camera()
        
        controller.add_module(ModuleType.VIDEO, camera_source=camera)
    
    def setup_all_modules(controller):
        """Helper to add all optional modules."""
        setup_video_module(controller, camera_type='sdk')
        controller.add_module(ModuleType.AUDIO)
        controller.add_module(ModuleType.OCR)
    
    # Use in main code
    controller = UnitreeGo2Controller(use_sdk=True)
    setup_all_modules(controller)

Troubleshooting Module Parameters
----------------------------------

**"Module requires camera_source" error**
    You forgot to pass the camera_source parameter to VIDEO module:

    .. code-block:: python

        # Wrong
        controller.add_module(ModuleType.VIDEO)
        
        # Correct
        camera = CameraSourceFactory.create_color_camera()
        controller.add_module(ModuleType.VIDEO, camera_source=camera)

**"Parameter not recognized" error**
    Check the parameter name matches the module's constructor. View the module documentation for the exact parameter name.

**"Module requires SDK mode"**
    Some modules (LIDAR, Input) only work with ``use_sdk=True``.

**"Module initialization failed"**
    Ensure all required parameters are of the correct type. For example, camera_source must be a ``CameraSource`` instance, not a string.

Next Steps
----------

- Read specific module documentation in :doc:`../api/index`
- Check the module's docstring in the API reference for exact parameter names
- See the :doc:`getting_started` guide for complete workflow examples
