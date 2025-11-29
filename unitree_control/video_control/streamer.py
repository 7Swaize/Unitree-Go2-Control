import os
from typing import Optional

import asyncio
import json
import queue
import threading
import socket

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from aiohttp import web


class OpenCVStreamTrack(VideoStreamTrack):
    def __init__(self, frame_queue):
        super().__init__()
        self._frame_queue = frame_queue

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self._frame_queue.get()

        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base

        return av_frame
    

class WebRTCStreamer:
    def __init__(self, host="0.0.0.0", port=8080):
        self._host = host
        self._port = port
        self._frame_queue = queue.Queue(maxsize=1)
        self._pc = None

        thread = threading.Thread(target=self._start_server, daemon=True)
        thread.start()

    def _send(self, frame):
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            
        self._frame_queue.put(frame)


    def _get_local_ip_address(self):
        s = None

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except:
            ip = "127.0.0.1"
        finally:
            if s is not None:
                s.close()

        return ip
    

    async def _offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        self._pc = RTCPeerConnection()
        self._pc.addTrack(OpenCVStreamTrack(self._frame_queue))

        await self._pc.setRemoteDescription(offer)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": self._pc.localDescription.sdp,
                "type": self._pc.localDescription.type
            })
        )
    

    async def _serve_html(self, request):
        path = os.path.join(os.path.dirname(__file__), "web", "viewer.html")
        return web.FileResponse(path)


    async def _run_server(self):
        app = web.Application()
        app.router.add_post("/offer", self._offer)
        app.router.add_get("/", self._serve_html)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        print(f"[WebRTC] Server running at http://{self._host}:{self._port}")
        await site.start()

        while True:
            await asyncio.sleep(3600)


    def _start_server(self):
        asyncio.run(self._run_server())