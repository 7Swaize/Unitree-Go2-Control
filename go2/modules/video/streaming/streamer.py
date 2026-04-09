import asyncio
import queue
import threading
import socket
import time

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiohttp import web
import cv2

from .stream_track import OpenCVStreamTrack
from .serve import HTML_CONTENT
    

# TURN for public conn? https://www.100ms.live/blog/webrtc-python-react#interactive-connectivity-establishment---ice
class WebRTCStreamer:
    """
    WebRTC streaming server for internal video broadcasting.

    Handles serving a video stream over a local WebRTC connection.
    This class uses:

        - `OpenCVStreamTrack` to pull frames from a queue
        - `aiortc` for WebRTC communication
        - `aiohttp` for HTTP endpoints and offer/answer negotiation
        - Asyncio and a background thread for the event loop

    Parameters
    ----------
    host : str
        The hostname or IP to bind the server (default "0.0.0.0").
    port : int
        The TCP port for the server (default 8080).

    Attributes
    ----------
    _frame_queue : queue.Queue
        Queue for frames to send to WebRTC clients.
    _pcs : set
        Set of active RTCPeerConnections.
    _loop : asyncio.AbstractEventLoop
        Event loop for running the server in a background thread.
    """
    def __init__(self, host="0.0.0.0", port=8080):
        self._host = host
        self._port = port
        self._frame_queue = queue.Queue(maxsize=10) 
        self._pcs = set()
        self._last_send_time = time.time()

        self._loop = None


    def _start_in_thread(self):
        """
        Start the WebRTC server in a daemon thread.

        Initializes a new asyncio event loop in a separate thread
        and runs the HTTP + WebRTC server asynchronously.
        """
        def _init_event_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_server())

        self._server_thread = threading.Thread(target=_init_event_loop, daemon=True)
        self._server_thread.start()


    def _send(self, frame):
        """
        Enqueue a video frame for streaming.

        Parameters
        ----------
        frame : numpy.ndarray
            The video frame to send. If the internal queue exceeds
            5 frames, the oldest frames are discarded.

        Notes
        -----
        - Non-blocking; frames are dropped if the queue is full.
        """
        if frame is None:
            return
            
        # clear queue if frames full
        while self._frame_queue.qsize() > 3:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
            
        self._frame_queue.put(frame, block=False)


    def _get_local_ip_address(self):
        """
        Determine the local LAN IP address for client connections.

        Returns
        -------
        str
            Local IP address (e.g., 192.168.x.x), or "127.0.0.1"
            if detection fails.
        """
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80)) 
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            if s:
                s.close()
        
        return ip


    def _get_port(self):
        """
        Get the TCP port used by the server.

        Returns
        -------
        int
            The port number.
        """
        return self._port


    async def _offer(self, request):  
        """
        Handle incoming SDP offers from clients.

        Parameters
        ----------
        request : aiohttp.web.Request
            The HTTP POST request containing the SDP offer.

        Returns
        -------
        aiohttp.web.Response
            JSON response containing the SDP answer.
        """
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self._pcs.add(pc)
        
        track = OpenCVStreamTrack(self._frame_queue)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            print(f"[WebRTC] Connection state: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                self._pcs.discard(pc)

        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            if pc.iceConnectionState in ["failed", "closed"]:
                await pc.close()
                self._pcs.discard(pc)

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })


    async def _serve_html(self, request):
        """
        Serve the WebRTC demo HTML page to clients.

        This page connects to the WebRTC server and displays
        video frames in the browser. For internal use only.
        """
        return web.Response(text=HTML_CONTENT, content_type="text/html")


    async def _run_server(self):
        """
        Run the aiohttp + WebRTC server asynchronously.

        Adds endpoints for "/offer" and "/" (HTML page) and keeps
        the server alive indefinitely.
        """
        app = web.Application()
        app.router.add_post("/offer", self._offer)
        app.router.add_get("/", self._serve_html)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        while True:
            await asyncio.sleep(3600)


    def _shutdown(self):
        """
        Shutdown all active connections and cleanup the server.

        Notes
        -----
        - Closes all RTCPeerConnections.
        - Cleans up the aiohttp runner.
        - Should be called before program exit.
        """
        async def _async_shutdown():
            futures = [pc.close() for pc in self._pcs]
            await asyncio.gather(*futures)
            self._pcs.clear()

            if hasattr(self, '_runner'):
                await self._runner.cleanup()

        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_async_shutdown(), self._loop)
