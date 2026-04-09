from .camera_source import CameraSource, OpenCVCameraSource, RealSenseDepthCameraSource, NativeCameraSource


class CameraSourceFactory:
    """
    Factory class for creating camera sources.

    ``CameraSourceFactory`` provides a simple, safe way for users to
    select which camera hardware to use without needing to understand
    low-level camera implementations.

    The returned objects are compatible with :class:`VideoModule` and
    should not be used directly.

    Notes
    -----
    Users should **not** instantiate camera source classes directly.
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
        """
        return NativeCameraSource()


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
        """
        return RealSenseDepthCameraSource()