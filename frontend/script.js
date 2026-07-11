document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    const skeletonImg = document.getElementById('skeleton-img');
    const skeletonPlaceholder = document.getElementById('skeleton-placeholder');
    const charDisplay = document.getElementById('detected-char');
    const sentenceDisplay = document.getElementById('sentence');
    const btnSpeak = document.getElementById('btn-speak');
    const btnClear = document.getElementById('btn-clear');
    
    const sugButtons = [
        document.getElementById('sug-0'),
        document.getElementById('sug-1'),
        document.getElementById('sug-2'),
        document.getElementById('sug-3')
    ];
    
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    let ws;
    let isStreaming = false;
    let frameInterval = null;

    // Health check — wait for backend before connecting
    async function waitForServer(maxAttempts = 30) {
        const protocol = window.location.protocol;
        const host = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
            ? 'localhost:8000'
            : window.location.host;
        const healthUrl = `${protocol}//${host}/health`;
        statusDot.className = "dot disconnected";

        for (let i = 1; i <= maxAttempts; i++) {
            statusText.innerText = `Waking up server... (${i}/${maxAttempts})`;
            try {
                const res = await fetch(healthUrl);
                if (res.ok) return true;
            } catch (e) {}
            await new Promise(r => setTimeout(r, 3000));
        }
        statusText.innerText = "Server unavailable. Please refresh.";
        return false;
    }

    // Setup webcam — use lower resolution for speed
    async function setupWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240, facingMode: "user" }
            });
            video.srcObject = stream;

            video.onloadedmetadata = async () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                isStreaming = true;

                const serverReady = await waitForServer();
                if (serverReady) connectWebSocket();
            };
        } catch (err) {
            console.error("Webcam error:", err);
            statusText.innerText = "Camera Access Denied";
            statusDot.className = "dot disconnected";
            alert("Please allow camera access for this application to work.");
        }
    }

    // WebSocket connection
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
            ? 'ws://localhost:8000/ws/predict'
            : `${protocol}//${window.location.host}/ws/predict`;

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connected");
            statusText.innerText = "Live";
            statusDot.className = "dot connected";

            // Send frames every 150ms — decoupled from server response
            if (frameInterval) clearInterval(frameInterval);
            frameInterval = setInterval(sendFrame, 150);
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'prediction' || data.type === 'state_update') {
                updateUI(data);
            }
        };

        ws.onclose = () => {
            console.log("WebSocket disconnected");
            statusText.innerText = "Disconnected — reconnecting...";
            statusDot.className = "dot disconnected";
            if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }

            setTimeout(async () => {
                const serverReady = await waitForServer();
                if (serverReady) connectWebSocket();
            }, 3000);
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
        };
    }

    // Send a single frame — low quality for speed
    function sendFrame() {
        if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const base64Img = canvas.toDataURL('image/jpeg', 0.5);
        ws.send(JSON.stringify({ type: 'frame', image: base64Img }));
    }

    // Update UI
    function updateUI(data) {
        if (data.skeleton) {
            skeletonImg.src = data.skeleton;
            skeletonImg.style.display = 'block';
            skeletonPlaceholder.style.display = 'none';
        }
        if (data.character !== undefined) charDisplay.innerText = data.character;
        if (data.sentence  !== undefined) sentenceDisplay.innerText = data.sentence;
        if (data.suggestions) {
            for (let i = 0; i < 4; i++) {
                sugButtons[i].innerText = data.suggestions[i] || " ";
            }
        }
    }

    // Speak button
    btnSpeak.addEventListener('click', () => {
        const text = sentenceDisplay.innerText.trim();
        if (text) {
            const utterance = new SpeechSynthesisUtterance(text.toLowerCase());
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    });

    // Clear button
    btnClear.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'clear' }));
        }
    });

    // Suggestion buttons
    sugButtons.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            if (btn.innerText.trim() && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'action', index }));
            }
        });
    });

    setupWebcam();
});
