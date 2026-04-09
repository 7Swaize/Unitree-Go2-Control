# AI GENERATED HTML - Shoutout Claude for this nice work

HTML_CONTENT = """<!DOCTYPE html>
<html>
<head>
    <title>Go2 Stream</title>
    <meta charset="UTF-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 24px;
        }

        h1 {
            font-size: 13px;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #555;
            margin-bottom: 16px;
        }

        #video-container {
            position: relative;
        }

        video {
            display: block;
            border: 1px solid #222;
            background: #000;
        }

        #overlay {
            position: absolute;
            top: 8px;
            left: 8px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(0,0,0,0.72);
            border: 1px solid #333;
            border-radius: 3px;
            padding: 3px 8px;
            font-size: 11px;
            letter-spacing: 0.08em;
        }

        .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #444;
            flex-shrink: 0;
        }
        .dot.connecting { background: #f5a623; animation: pulse 1.2s infinite; }
        .dot.connected   { background: #4caf50; }
        .dot.failed      { background: #f44336; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.3; }
        }

        #log {
            margin-top: 16px;
            width: 640px;
            background: #0f0f0f;
            border: 1px solid #1e1e1e;
            border-radius: 4px;
            padding: 10px 12px;
            font-size: 11px;
            color: #555;
            max-height: 180px;
            overflow-y: auto;
            line-height: 1.6;
        }

        #stream-links {
            margin-top: 12px;
            font-size: 11px;
            color: #444;
            letter-spacing: 0.05em;
        }

        #stream-links a {
            color: #2196f3;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <h1>Unitree Go2 &mdash; Camera Stream</h1>

    <div id="video-container">
        <video id="video" autoplay playsinline muted width="640" height="480"></video>
        <div id="overlay">
            <div class="badge">
                <div class="dot connecting" id="status-dot"></div>
                <span id="status-text">Connecting via WebRTC…</span>
            </div>
            <div class="badge" id="stats-badge" style="display:none">
                <span id="stats-text">—</span>
            </div>
        </div>
    </div>

    <div id="log">Initialising…</div>

    <script>
    (function () {
        const dot      = document.getElementById('status-dot');
        const statusEl = document.getElementById('status-text');
        const logEl    = document.getElementById('log');
        const statsEl  = document.getElementById('stats-text');
        const statsBadge = document.getElementById('stats-badge');

        function log(msg) {
            const ts = new Date().toLocaleTimeString('en', { hour12: false });
            const line = document.createElement('div');
            line.textContent = `[${ts}] ${msg}`;
            logEl.insertBefore(line, logEl.firstChild);
        }

        function setStatus(state, text) {
            dot.className = 'dot ' + state;
            statusEl.textContent = text;
        }

        window.onerror = (msg, _url, line) => log(`ERROR ${line}: ${msg}`);

        async function startWebRTC() {
            log('Creating RTCPeerConnection…');

            const pc = new RTCPeerConnection({ iceServers: [] });

            pc.onconnectionstatechange = () => {
                log(`Connection: ${pc.connectionState}`);
                if (pc.connectionState === 'connected') {
                    setStatus('connected', 'WebRTC — live');
                    startStatsPoller(pc);
                } else if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
                    setStatus('failed', 'Disconnected — reload to retry');
                }
            };

            pc.oniceconnectionstatechange = () => log(`ICE: ${pc.iceConnectionState}`);
            pc.onicegatheringstatechange  = () => log(`Gathering: ${pc.iceGatheringState}`);

            pc.ontrack = e => {
                log(`Track received: ${e.track.kind}`);
                const video = document.getElementById('video');
                video.srcObject = e.streams[0] || new MediaStream([e.track]);

                // Chrome 107+: collapse the jitter buffer to near-zero
                const receiver = pc.getReceivers().find(r => r.track.kind === 'video');
                if (receiver && receiver.jitterBufferTarget !== undefined) {
                    receiver.jitterBufferTarget = 0;
                    log('jitterBufferTarget set to 0');
                }
            };

            // ── Offer ──────────────────────────────────────────────────────────
            const offer = await pc.createOffer({
                offerToReceiveVideo: true,
                offerToReceiveAudio: false,
            });

            // Patch SDP: request low-latency VP8 with bounded bitrate
            offer.sdp = patchSDP(offer.sdp);
            await pc.setLocalDescription(offer);
            log('Local description set');

            const resp = await fetch('/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
            });

            if (!resp.ok) throw new Error(`Server ${resp.status}: ${await resp.text()}`);

            const answer = await resp.json();
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
            log('Remote description set — waiting for ICE…');
        }

        // ── SDP patch: prefer VP8, cap bitrate, disable B-frames ──────────────
        function patchSDP(sdp) {
            // Add bandwidth annotation
            sdp = sdp.replace(/a=mid:video/g, 'a=mid:video\\r\\nb=AS:4000');

            // Inject low-latency VP8 fmtp if present
            sdp = sdp.replace(
                /(a=rtpmap:\\d+ VP8\\/90000\\r\\n)/g,
                '$1a=fmtp:{pt} x-google-start-bitrate=1500;x-google-max-bitrate=4000\\r\\n'
            );

            return sdp;
        }

        // ── Live stats badge ───────────────────────────────────────────────────
        function startStatsPoller(pc) {
            statsBadge.style.display = 'flex';
            setInterval(async () => {
                const reports = await pc.getStats();
                reports.forEach(r => {
                    if (r.type === 'inbound-rtp' && r.kind === 'video') {
                        const fps    = (r.framesPerSecond || 0).toFixed(0);
                        const kbps   = ((r.bytesReceived || 0) * 8 / 1000 / (r.timestamp / 1000)).toFixed(0);
                        const jitter = ((r.jitter || 0) * 1000).toFixed(1);
                        statsEl.textContent = `${fps} fps · ${jitter} ms jitter`;
                    }
                });
            }, 1000);
        }

        // ── Boot ───────────────────────────────────────────────────────────────
        setStatus('connecting', 'Connecting via WebRTC…');
        startWebRTC().catch(err => {
            log(`FATAL: ${err.message}`);
            setStatus('failed', `Error: ${err.message}`);
        });
    })();
    </script>
</body>
</html>"""