from src.video_control.camera_source import CameraSource, OpenCVCameraSource, RealSenseDepthCamera, SDKCameraSource


class CameraSourceFactory:
    """
    Factory class for creating camera sources.

    ``CameraSourceFactory`` provides a simple, safe way for students to
    select which camera hardware to use without needing to understand
    low-level camera implementations.

    The returned objects are compatible with :class:`VideoModule` and
    should not be used directly.

    Notes
    -----
    Students should **not** instantiate camera source classes directly.
    Always use this factory instead.
    """


    @staticmethod
    def create_sdk_camera() -> CameraSource:
        """
        Create a camera source backed by the robot's internal camera.

        This camera is typically used when running code directly on
        the robot hardware.

        Returns
        -------
        CameraSource
            A camera source compatible with :class:`VideoModule`.

        Example
        -------
        >>> from src.core.unitree_control_core import UnitreeGo2Controller
        >>> from src.core.module_registry import ModuleType
        >>> from src.video_control.camera_source_factory import CameraSourceFactory
        >>>
        >>> unitree_controller = UnitreeGo2Controller(sdk_enabled=True)
        >>> unitree_controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_sdk_camera()) # Add video module with SDK camera
        """
        return SDKCameraSource()


    @staticmethod
    def create_opencv_camera(camera_index: int = 0) -> CameraSource:
        """
        Create a camera source using OpenCV's ``VideoCapture``.

        This is the recommended option for:
            - USB webcams
            - Laptop cameras
            - Development on personal machines

        Parameters
        ----------
        camera_index : int, optional
            Index of the OpenCV camera (default is 0).

        Returns
        -------
        CameraSource
            A camera source compatible with :class:`VideoModule`.

        Example
        -------
        >>> from src.core.unitree_control_core import UnitreeGo2Controller
        >>> from src.core.module_registry import ModuleType
        >>> from src.video_control.camera_source_factory import CameraSourceFactory
        >>>
        >>> unitree_controller = UnitreeGo2Controller(sdk_enabled=False)
        >>> unitree_controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_opencv_camera(0)) # Add video module with default webcam
        """
        return OpenCVCameraSource(camera_index)


    @staticmethod
    def create_depth_camera() -> CameraSource:
        """
        Create an RGB-D camera source using an Intel RealSense device.

        This camera provides both color and depth frames. The returned
        camera source automatically aligns depth data to the color frame.

        Returns
        -------
        CameraSource
            A camera source compatible with :class:`VideoModule`.

        Notes
        -----
        Requires an Intel RealSense camera and the RealSense SDK.

        Example
        -------
        >>> from src.core.unitree_control_core import UnitreeGo2Controller
        >>> from src.core.module_registry import ModuleType
        >>> from src.video_control.camera_source_factory import CameraSourceFactory
        >>>
        >>> unitree_controller = UnitreeGo2Controller(sdk_enabled=True)
        >>> unitree_controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_depth_camera()) # Add video module with depth camera
        """
        return RealSenseDepthCamera()