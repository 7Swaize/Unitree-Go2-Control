"""WebRTC Streaming Interface"""

import threading
import socket
from typing import Optional
import numpy as np


class WebRTCStreamer:
    """
    WebRTC streaming server for video frames.
    
    This is a simplified interface stub for WebRTC streaming.
    The full implementation would use a WebRTC library.
    """
    
    def __init__(self, port: int = 8080):
        """Initialize the streaming server."""
        self._port = port
        self._running = False
        self._thread = None
        self._local_ip = self._get_local_ip()

    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            # Get local IP by connecting to external address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def start_in_thread(self) -> None:
        """Start the streaming server in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def _run_server(self) -> None:
        """Run the streaming server."""
        # Simplified: in full implementation would set up WebRTC endpoint
        print(f"[Video] Streaming server started at {self._local_ip}:{self._port}")

    def send(self, frame: np.ndarray) -> None:
        """Send a frame to connected clients."""
        if not self._running:
            raise RuntimeError("Stream server not running")
        # Simplified: in full implementation would encode and send frame

    def get_local_ip_address(self) -> str:
        """Get the local IP address where the stream is hosted."""
        return self._local_ip

    def get_port(self) -> int:
        """Get the port number of the stream server."""
        return self._port

    def shutdown(self) -> None:
        """Shut down the streaming server."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
