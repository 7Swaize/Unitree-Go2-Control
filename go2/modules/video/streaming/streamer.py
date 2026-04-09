import asyncio
import threading
import socket

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCRtpSender
from aiohttp import web

from .stream_config import StreamConfig
from .stream_track import OpenCVStreamTrack
from .serve import HTML_CONTENT
    

# TURN for public conn? https://www.100ms.live/blog/webrtc-python-react#interactive-connectivity-establishment---ice
class WebRTCStreamer:
    def __init__(self, stream_config: StreamConfig):
        self._stream_config = stream_config
        self._track = OpenCVStreamTrack(stream_config)
 
        self._pcs: set[RTCPeerConnection] = set()
        self._loop: asyncio.AbstractEventLoop = None
        self._server_thread: threading.Thread = None


    def _start_in_thread(self):
        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_server())

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()


    def _send(self, frame):
        if frame is None:
            return
        self._track.push_frame(frame)


    def _get_local_ip_address(self):
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
        return self._stream_config.port


    async def _serve_html(self, request: web.Request) -> web.Response:
        return web.Response(text=HTML_CONTENT, content_type="text/html")

    async def _offer(self, request):  
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self._pcs.add(pc)
        
        transceiver = pc.addTransceiver(self._track, direction="sendonly")
        try:
            caps = RTCRtpSender.getCapabilities("video")
            vp8_codecs = [c for c in caps.codecs if "VP8" in c.mimeType.upper()]
            if vp8_codecs:
                transceiver.setCodecPreferences(vp8_codecs)
        except Exception:
            pass # Just best effort


        @pc.on("connectionstatechange")
        async def _on_state():
            print(f"[WebRTC] {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self._pcs.discard(pc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
 
        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        })


    async def _run_server(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._serve_html)
        app.router.add_post("/offer", self._offer)
 
        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner
 
        site = web.TCPSite(runner, self._stream_config.host, self._stream_config.port)
        await site.start()
          
        while True:
            await asyncio.sleep(3600)


    def _shutdown(self) -> None:
        async def _async_shutdown():
            await asyncio.gather(*[pc.close() for pc in self._pcs], return_exceptions=True)
            self._pcs.clear()
            if hasattr(self, "_runner"):
                await self._runner.cleanup()
 
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_async_shutdown(), self._loop)
