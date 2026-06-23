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

    // --- Health check: wait for backend to wake up before connecting ---
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
                if (res.ok) {
                    console.log("Server is up!");
                    return true;
                }
            } catch (e) {
                // Server not ready yet, keep retrying
            }
            await new Promise(r => setTimeout(r, 3000)); // wait 3 seconds between tries
        }

        statusText.innerText = "Server unavailable. Please refresh.";
        return false;
    }

    // 1. Setup Webcam
    async function setupWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: "user" }
            });
            video.srcObject = stream;
            
            video.onloadedmetadata = async () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                isStreaming = true;

                // Wait for server before opening WebSocket
                const serverReady = await waitForServer();
                if (serverReady) {
                    connectWebSocket();
                }
            };
        } catch (err) {
            console.error("Error accessing webcam: ", err);
            statusText.innerText = "Camera Access Denied";
            statusDot.className = "dot disconnected";
            alert("Please allow camera access for this application to work.");
        }
    }

    // 2. Setup WebSocket
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
            ? `ws://localhost:8000/ws/predict`
            : `${protocol}//${window.location.host}/ws/predict`;
            
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connected");
            statusText.innerText = "Live";
            statusDot.className = "dot connected";
            sendFrames();
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'prediction' || data.type === 'state_update') {
                updateUI(data);
            }
            
            if (data.type === 'prediction') {
                requestAnimationFrame(sendFrames);
            }
        };

        ws.onclose = () => {
            console.log("WebSocket disconnected");
            statusText.innerText = "Disconnected — reconnecting...";
            statusDot.className = "dot disconnected";
            // Retry the full health-check + reconnect cycle
            setTimeout(async () => {
                const serverReady = await waitForServer();
                if (serverReady) connectWebSocket();
            }, 3000);
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
        };
    }

    // 3. Send frames to backend
    function sendFrames() {
        if (!isStreaming || ws.readyState !== WebSocket.OPEN) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const base64Img = canvas.toDataURL('image/jpeg', 0.8);
        
        ws.send(JSON.stringify({
            type: 'frame',
            image: base64Img
        }));
    }

    // 4. Update UI with backend data
    function updateUI(data) {
        if (data.skeleton) {
            skeletonImg.src = data.skeleton;
            skeletonImg.style.display = 'block';
            skeletonPlaceholder.style.display = 'none';
        }

        if (data.character !== undefined) {
            charDisplay.innerText = data.character;
        }

        if (data.sentence !== undefined) {
            sentenceDisplay.innerText = data.sentence;
        }

        if (data.suggestions) {
            for (let i = 0; i < 4; i++) {
                sugButtons[i].innerText = data.suggestions[i] || " ";
            }
        }
    }

    // 5. Button Event Listeners
    btnSpeak.addEventListener('click', () => {
        const textToSpeak = sentenceDisplay.innerText.trim();
        if (textToSpeak !== "") {
            const utterance = new SpeechSynthesisUtterance(textToSpeak.toLowerCase());
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    });

    btnClear.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'clear' }));
        }
    });

    sugButtons.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            if (btn.innerText.trim() !== "" && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'action',
                    index: index
                }));
            }
        });
    });

    // Start everything
    setupWebcam();
});
