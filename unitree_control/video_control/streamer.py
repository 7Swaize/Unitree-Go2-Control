import os
from typing import Optional

import asyncio
import json
import queue
import threading
import socket
import time

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from aiohttp import web
import numpy as np


# command to install rtc: pip install aiortc opencv-python

class OpenCVStreamTrack(VideoStreamTrack):
    def __init__(self, frame_queue):
        super().__init__()
        self._frame_queue = frame_queue
        self._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._frame_count = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = None
        while True:
            try:
                frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        if frame is not None:
            self._last_frame = frame
            self._frame_count += 1

        frame = self._last_frame.copy()

        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base
        return av_frame
    

# TURN for public conn? https://www.100ms.live/blog/webrtc-python-react#interactive-connectivity-establishment---ice
class WebRTCStreamer:
    def __init__(self, host="0.0.0.0", port=8080):
        self._host = host
        self._port = port
        self._frame_queue = queue.Queue(maxsize=10) 
        self._pcs = set()
        self._last_send_time = time.time()

        self._loop = None


    def start_in_thread(self):
        """Start the WebRTC server in a background thread"""
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_server())
        
        self._server_thread = threading.Thread(target=run, daemon=True)
        self._server_thread.start()
        time.sleep(0.5)


    def send(self, frame):
        """Send a frame to the WebRTC stream"""
        if frame is None:
            return
            
        # clear queue if frames full
        while self._frame_queue.qsize() > 5:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
            
        self._frame_queue.put(frame, block=False)


    def get_local_ip_address(self):
        """
        Returns the LAN IP address that clients can connect to.
        Works even if the robot has multiple interfaces.
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
    

    async def _offer(self, request):     
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


    # AI GENERATED HTML - Shoutout Claude for this nice work
    async def _serve_html(self, request):
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>WebRTC Stream</title>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            background: #f0f0f0;
        }
        h1 {
            color: #333;
        }
        video {
            border: 2px solid #333;
            border-radius: 8px;
            background: #000;
        }
        #status {
            margin-top: 10px;
            padding: 10px;
            border-radius: 4px;
            font-weight: bold;
        }
        .connecting { background: #fff3cd; color: #856404; }
        .connected { background: #d4edda; color: #155724; }
        .failed { background: #f8d7da; color: #721c24; }
        #debug {
            margin-top: 20px;
            padding: 10px;
            background: #fff;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            max-width: 640px;
            text-align: left;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <h1>WebRTC Stream</h1>
    <video id="video" autoplay playsinline muted width="640" height="480"></video>
    <div id="status" class="connecting">Connecting...</div>
    <div id="debug">Starting JavaScript...</div>

    <script>
    (function() {
        const statusEl = document.getElementById('status');
        const debugEl = document.getElementById('debug');
        
        function log(msg) {
            const timestamp = new Date().toLocaleTimeString();
            const logMsg = `[${timestamp}] ${msg}`;
            console.log(logMsg);
            debugEl.textContent = logMsg + '\\n' + debugEl.textContent.split('\\n').slice(0, 15).join('\\n');
        }

        // Catch any uncaught errors
        window.onerror = function(msg, url, line, col, error) {
            log(`ERROR: ${msg} at ${line}:${col}`);
            if (error) log(`Stack: ${error.stack}`);
            return false;
        };

        log('Script loaded successfully');

        async function start() {
            try {
                log('Starting WebRTC connection...');
                
                // Check if browser supports WebRTC
                if (!window.RTCPeerConnection) {
                    throw new Error('WebRTC not supported in this browser');
                }
                
                log('Creating peer connection...');
                const pc = new RTCPeerConnection({
                    iceServers: []  // Local connection, no STUN/TURN needed
                });

                pc.onconnectionstatechange = () => {
                    log(`Connection state: ${pc.connectionState}`);
                    if (pc.connectionState === 'connected') {
                        statusEl.textContent = 'Connected';
                        statusEl.className = 'connected';
                    } else if (pc.connectionState === 'failed') {
                        statusEl.textContent = 'Connection failed';
                        statusEl.className = 'failed';
                    } else if (pc.connectionState === 'disconnected') {
                        statusEl.textContent = 'Disconnected';
                        statusEl.className = 'failed';
                    }
                };

                pc.oniceconnectionstatechange = () => {
                    log(`ICE state: ${pc.iceConnectionState}`);
                };

                pc.onicegatheringstatechange = () => {
                    log(`ICE gathering: ${pc.iceGatheringState}`);
                };

                pc.onicecandidate = (event) => {
                    if (event.candidate) {
                        log(`ICE candidate: ${event.candidate.candidate.substring(0, 50)}...`);
                    } else {
                        log('ICE gathering complete');
                    }
                };

                // Receive remote video from server
                pc.ontrack = e => {
                    log(`Received ${e.track.kind} track`);
                    const video = document.getElementById('video');
                    
                    if (e.streams && e.streams[0]) {
                        video.srcObject = e.streams[0];
                        log('Video stream attached');
                    } else {
                        log('WARNING: No stream in track event');
                    }
                    
                    video.onloadedmetadata = () => {
                        log(`Video size: ${video.videoWidth}x${video.videoHeight}`);
                    };
                    
                    video.onplay = () => {
                        log('Video started playing!');
                    };
                    
                    video.onerror = (e) => {
                        log(`Video error: ${e.message || 'unknown'}`);
                    };
                };

                // Create offer to receive video
                log('Creating offer...');
                const offer = await pc.createOffer({ 
                    offerToReceiveVideo: true,
                    offerToReceiveAudio: false
                });
                
                log('Setting local description...');
                await pc.setLocalDescription(offer);
                log('Local description set');

                // Send SDP offer to Python server
                log('Sending offer to server...');
                const resp = await fetch("/offer", {
                    method: "POST",
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    }),
                    headers: { "Content-Type": "application/json" }
                });

                if (!resp.ok) {
                    const errorText = await resp.text();
                    throw new Error(`Server error ${resp.status}: ${errorText}`);
                }

                log('Server responded successfully');
                const answer = await resp.json();
                log(`Received answer (${answer.type})`);
                
                log('Setting remote description...');
                await pc.setRemoteDescription(new RTCSessionDescription(answer));
                log('Remote description set - connection should establish...');
                
            } catch (error) {
                log(`FATAL ERROR: ${error.message}`);
                if (error.stack) log(`Stack: ${error.stack}`);
                statusEl.textContent = 'Error: ' + error.message;
                statusEl.className = 'failed';
            }
        }

        // Start immediately when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', start);
        } else {
            start();
        }
    })();
    </script>
</body>
</html>"""
        return web.Response(text=html_content, content_type="text/html")


    async def _run_server(self):
        app = web.Application()
        app.router.add_post("/offer", self._offer)
        app.router.add_get("/", self._serve_html)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        
        await site.start()

        local_ip = self.get_local_ip_address()
        print(f"[WebRTC] Server started successfully!")
        print(f" Local Network: http://{local_ip}:{self._port}")

        while True:
            await asyncio.sleep(3600)


    # VERY IMPORTANT TO IMPLEMENT THIS SOON
    def _shutdown(self):
        pass