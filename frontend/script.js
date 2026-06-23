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

    // 1. Setup Webcam
    async function setupWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: "user" }
            });
            video.srcObject = stream;
            
            // Wait for video to load to set canvas size
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                isStreaming = true;
                connectWebSocket(); // Connect WS only after video is ready
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
        // Use wss:// in production, ws:// locally
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // For local development, assume backend is at localhost:8000
        const wsUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
            ? `ws://localhost:8000/ws/predict`
            : `${protocol}//${window.location.host}/ws/predict`; // Production assumption
            
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connected");
            statusText.innerText = "Live";
            statusDot.className = "dot connected";
            
            // Start sending frames
            sendFrames();
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'prediction' || data.type === 'state_update') {
                updateUI(data);
            }
            
            // Ask for the next frame ONLY after the server has finished processing the current one
            if (data.type === 'prediction') {
                requestAnimationFrame(sendFrames);
            }
        };

        ws.onclose = () => {
            console.log("WebSocket disconnected");
            statusText.innerText = "Disconnected";
            statusDot.className = "dot disconnected";
            
            // Attempt to reconnect after 3 seconds
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
        };
    }

    // 3. Send frames to backend
    function sendFrames() {
        if (!isStreaming || ws.readyState !== WebSocket.OPEN) return;

        // Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get base64 string with high enough quality to prevent MediaPipe tracking loss
        const base64Img = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to backend
        ws.send(JSON.stringify({
            type: 'frame',
            image: base64Img
        }));
        
        // Notice: setTimeout is removed. The next frame is triggered by ws.onmessage
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
            // Replace double spaces with visual space for clarity if needed
            sentenceDisplay.innerText = data.sentence;
        }

        if (data.suggestions) {
            for (let i = 0; i < 4; i++) {
                sugButtons[i].innerText = data.suggestions[i] || " ";
            }
        }
    }

    // 5. Button Event Listeners
    
    // Speak using Web Speech API
    btnSpeak.addEventListener('click', () => {
        const textToSpeak = sentenceDisplay.innerText.trim();
        if (textToSpeak !== "") {
            // Use toLowerCase() so the browser reads it as a full word instead of spelling an acronym
            const utterance = new SpeechSynthesisUtterance(textToSpeak.toLowerCase());
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    });

    // Clear string
    btnClear.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'clear' }));
        }
    });

    // Suggestion buttons
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
