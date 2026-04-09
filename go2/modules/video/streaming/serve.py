# AI GENERATED HTML - Shoutout Claude for this nice work

HTML_CONTENT = """<!DOCTYPE html>
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