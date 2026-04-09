from dataclasses import dataclass


@dataclass
class StreamConfig:
    width: int = 640 #: Width of the broadcast stream in pixels (default 640).
    height: int = 480 #: Height of the broadcast stream in pixels (default 480).
    fps: int = 30 #: Target broadcast frame rate (default 30).
    host: str = "0.0.0.0" #: The hostname or IP to bind the server (default "0.0.0.0").
    port: int = 8080 #: HTTP port for the stream server (default 8080).
