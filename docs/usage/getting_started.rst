Getting Started with Unitree Go2 Control
==========================================

Welcome! This guide will help you get up and running with the Unitree Go2 robot control system.

Quick Start
-----------

The entire Unitree Go2 control system is accessed through a single controller object:

.. code-block:: python

    from src.core.unitree_control_core import UnitreeGo2Controller

    # Create a controller to interact with the robot
    controller = UnitreeGo2Controller(use_sdk=True)

    # Default modules are automatically available
    controller.movement.stand_up()

    # Always clean up when done
    controller.safe_shutdown()

That's it! You now have access to the core functionality.

Understanding Default vs Optional Modules
------------------------------------------

The controller uses a modular design with two categories:

**Default Modules** (Always Available)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are automatically instantiated when you create the controller:

- **Movement Module** - Control locomotion and posture
- **Input Module** (SDK mode only) - Handle gamepad/joystick input

You can use them immediately without any additional setup.

**Optional Modules** (Add as Needed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need additional functionality, explicitly add modules using ``add_module()``:

- **Video Control** - Access camera feeds and streaming
- **Audio Control** - Play sounds and manage audio
- **LIDAR Control** - Process distance sensor data
- **OCR Control** - Optical character recognition
- **ArUco** - ArUco marker detection

Adding an optional module is simple:

.. code-block:: python

    from src.core.module_registry import ModuleType
    from src.video_control.camera_source_factory import CameraSourceFactory

    controller = UnitreeGo2Controller(use_sdk=True, CameraSourceFactory.create_opencv_camera(0))
    
    # Add the video module
    controller.add_module(ModuleType.VIDEO, )
    
    # Now you can use it
    status, frame = controller.video.get_frame()
    
    controller.safe_shutdown()

Available Modules Reference
---------------------------

===================== ================================ ===============================
Module                Purpose                          Category
===================== ================================ ===============================
**Movement**          Control locomotion and posture   Default
**Input Control**     Handle gamepad/joystick input    Default (SDK only)
**Video Control**     Access camera feeds              Optional
**Audio Control**     Play sounds and manage audio     Optional
**LIDAR Control**     Process distance sensor data     Optional
**OCR Control**       Optical character recognition    Optional
**ArUco**             ArUco marker detection           Optional
===================== ================================ ===============================

Combining Modules: Complete Examples
-------------------------------------

Example 1: Simple Movement (Using Default Modules)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from src.core.unitree_control_core import UnitreeGo2Controller

    controller = UnitreeGo2Controller(use_sdk=True)
    
    try:
        controller.movement.stand_up()
        time.sleep(1)
        controller.movement.move(x=0.5)  # Move forward
        time.sleep(2)
    finally:
        controller.safe_shutdown()

Example 2: Movement + Video (Adding Optional Modules)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from src.core.unitree_control_core import UnitreeGo2Controller
    from src.core.module_registry import ModuleType
    from src.video_control.camera_source_factory import CameraSourceFactory

    controller = UnitreeGo2Controller(use_sdk=True)
    
    # Add video module with a camera source
    camera = CameraSourceFactory.create_sdk_camera()
    controller.add_module(ModuleType.VIDEO, camera_source=camera)
    
    try:
        controller.movement.stand_up()
        time.sleep(1)
        
        # Capture a frame from the camera
        status, frame = controller.video.get_frames()
        if status == 0:
            print(f"Got frame with shape: {frame.shape}")
        
    finally:
        controller.safe_shutdown()

Example 3: Input-Based Control with Video Feedback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from src.core.unitree_control_core import UnitreeGo2Controller
    from src.core.module_registry import ModuleType
    from src.controller_input_control.input_signal import InputSignal
    from src.video_control.camera_source_factory import CameraSourceFactory

    controller = UnitreeGo2Controller(use_sdk=True)
    
    # Add video module
    camera = CameraSourceFactory.create_sdk_camera()
    controller.add_module(ModuleType.VIDEO, camera_source=camera)
    
    # Input module is already available by default in SDK mode
    
    try:
        controller.movement.stand_up()
        time.sleep(1)
        
        def on_button_x(state):
            """Handle button X press."""
            controller.movement.move(amount_x=0.5)  # Move forward
        
        def on_button_b(state):
            """Handle button B press."""
            controller.movement.stand_down()
        
        # Register input callbacks
        controller.input.register_callback(InputSignal.BUTTON_X, on_button_x)
        controller.input.register_callback(InputSignal.BUTTON_B, on_button_b)
        
        # Keep the program running to process callbacks
        while True:
            status, frame = controller.video.get_frames()
            if status == 0:
                # Process frame for visual feedback
                pass
            
            time.sleep(0.01)
            
    finally:
        controller.safe_shutdown()

Documentation Organization
----------------------------

The documentation is organized into sections to help you find what you need:

**Core & Defaults** 
    Start here to understand the main controller and default modules available immediately.

**Optional Modules**
    Add these when you need specific functionality like vision or audio.

**Architecture & Patterns**
    Learn about the state machine approach for building complex behaviors.

Module Usage Patterns
---------------------

Pattern 1: One-Shot Action (Minimal Setup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    controller = UnitreeGo2Controller(use_sdk=True)
    controller.movement.stand_up()
    controller.safe_shutdown()


Pattern 2: Continuous Loop with Default Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from src.core.unitree_control_core import UnitreeGo2Controller

    controller = UnitreeGo2Controller(use_sdk=True)

    controller.movement.stand_up()
    time.sleep(1)

    try:
        while True:
            # Move forward continuously
            controller.movement.move(1.0)
            time.sleep(0.02)
    finally:
        controller.movement.stop()
        controller.safe_shutdown()


Pattern 3: State Machine Approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more complex behaviors, you can optionally use the state machine pattern.
This allows you to encapsulate movement logic, transitions, and sensor processing
inside reusable state classes.

.. code-block:: python

    import time
    from src.states.dog_state_abstract import DogStateAbstract
    from src.core.unitree_control_core import UnitreeGo2Controller

    class MoveForwardState(DogStateAbstract):
        def __init__(self, controller):
            super().__init__(controller)

        def execute(self):
            # Implement additional state logic here (conditions, transitions, etc.)
            while True:
                self.unitree_controller.movement.move(1.0)
                time.sleep(0.02)

    controller = UnitreeGo2Controller(use_sdk=True)
    controller.movement.stand_up()
    time.sleep(1)

    state = MoveForwardState(controller)

    try:
        state.execute()
    finally:
        controller.movement.stop()
        controller.safe_shutdown()


Tips for Getting Started
------------------------

1. **Add modules as needed**
   Avoid loading modules you do not actively use. Unnecessary modules may increase
   resource usage or introduce unintended side effects.

2. **Always use ``try``–``finally`` blocks**
   Wrap your main robot control logic inside a ``try``–``finally`` block to ensure
   that cleanup code always executes. This guarantees the robot is safely stopped
   and shut down even if an exception, runtime error, or user interrupt occurs.

3. **Always call ``safe_shutdown()``**
   Calling ``safe_shutdown()`` ensures proper cleanup of hardware resources and
   communication channels. This is especially important because emergency stops
   (such as pressing ``Ctrl+C``) raise a ``KeyboardInterrupt`` exception, which
   would otherwise bypass normal shutdown logic.

4. **Test incrementally**
   Build complex behaviors step-by-step. Validate each movement or state in isolation
   before combining them into higher-level autonomy.

5. **Use simulation first**
   Start with ``use_sdk=False`` to test logic without physical hardware. This allows
   faster iteration and reduces the risk of unintended robot motion.


Note on Module Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Some modules require additional parameters when adding them. For example, the video module needs a camera source. Check the module documentation and the :doc:`module_parameters` guide for details on what each module requires.

Next Steps
----------

- Read the :doc:`../api/core` documentation for detailed API reference
- Choose a module to master (e.g., Movement or Video Control)
- Check out specific module documentation in the :doc:`../api/index`
- For advanced behaviors, see the :doc:`../api/states` documentation

Troubleshooting
---------------

**"Module not loaded" error?**
    Make sure you called ``add_module()`` before accessing an optional module.

**Hardware errors?**
    Check that ``use_sdk=True`` is set and hardware is connected.

**Input module not available?**
    The input module only loads in SDK mode (``use_sdk=True``).

**State validation issues?**
    See the :doc:`../api/states` documentation.
