import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import math
import cv2
import base64
import json
import asyncio
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
from spellchecker import SpellChecker
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/launch")
async def launch_app():
    return {"status": "success", "message": "Redirecting to web app"}

# ── Load TFLite model ────────────────────────────────────────────────────────
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded.")

# ── Load spell checker ───────────────────────────────────────────────────────
print("Loading spell checker...")
ddd = SpellChecker()
print("Spell checker loaded.")

# ── Hand detectors ───────────────────────────────────────────────────────────
print("Initialising hand detectors...")
hd  = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
offset = 29
print("All components loaded. Server ready.")


def run_inference(white_img):
    """Run TFLite inference on a 400x400x3 image. Returns prob array."""
    inp = white_img.reshape(1, 400, 400, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0].copy()


def safe_find_hands(detector, image, **kwargs):
    """Handles both old (list) and new (tuple) cvzone return formats."""
    result = detector.findHands(image, **kwargs)
    if isinstance(result, tuple):
        return result[0]   # new cvzone: (hands, img)
    return result          # old cvzone: hands


# ── Session class ────────────────────────────────────────────────────────────
class SignLanguageSession:
    def __init__(self):
        self.ct            = {i: 0 for i in ascii_uppercase}
        self.ct['blank']   = 0
        self.blank_flag    = 0
        self.space_flag    = False
        self.next_flag     = True
        self.prev_char     = ""
        self.count         = -1
        self.ten_prev_char = [" "] * 10
        self.str_sentence  = " "
        self.ccc           = 0
        self.word          = " "
        self.current_symbol= "C"
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        self.pts   = None

    def distance(self, x, y):
        return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

    def process_frame(self, frame_b64):
        header, encoded = (frame_b64.split(",", 1) if "," in frame_b64
                           else ("", frame_b64))
        nparr    = np.frombuffer(base64.b64decode(encoded), np.uint8)
        cv2image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2image = cv2.flip(cv2image, 1)

        skeleton_b64 = None

        if cv2image is None or cv2image.size == 0:
            return self._state(skeleton_b64)

        hands = safe_find_hands(hd, cv2image, draw=False, flipType=True)

        if not hands:
            return self._state(skeleton_b64)

        hand = hands[0] if isinstance(hands, list) else hands
        try:
            x, y, w, h = hand['bbox']
        except Exception as e:
            print(f"bbox error: {e}")
            return self._state(skeleton_b64)

        h_img, w_img, _ = cv2image.shape
        y1 = max(0, y - offset)
        y2 = min(h_img, y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(w_img, x + w + offset)
        image = cv2image[y1:y2, x1:x2]

        if image.size == 0:
            return self._state(skeleton_b64)

        white = np.ones((400, 400, 3), dtype=np.uint8) * 255
        handz = safe_find_hands(hd2, image, draw=False, flipType=True)
        self.ccc += 1

        if not handz:
            return self._state(skeleton_b64)

        hand2   = handz[0] if isinstance(handz, list) else handz
        try:
            self.pts = hand2['lmList']
        except Exception as e:
            print(f"lmList error: {e}")
            return self._state(skeleton_b64)

        os_x  = ((400 - w) // 2) - 15
        os1_y = ((400 - h) // 2) - 15

        try:
            p = self.pts
            # Draw finger bones
            for t in range(0, 4):
                cv2.line(white, (p[t][0]+os_x,   p[t][1]+os1_y),
                                (p[t+1][0]+os_x, p[t+1][1]+os1_y), (0,255,0), 3)
            for t in range(5, 8):
                cv2.line(white, (p[t][0]+os_x,   p[t][1]+os1_y),
                                (p[t+1][0]+os_x, p[t+1][1]+os1_y), (0,255,0), 3)
            for t in range(9, 12):
                cv2.line(white, (p[t][0]+os_x,   p[t][1]+os1_y),
                                (p[t+1][0]+os_x, p[t+1][1]+os1_y), (0,255,0), 3)
            for t in range(13, 16):
                cv2.line(white, (p[t][0]+os_x,   p[t][1]+os1_y),
                                (p[t+1][0]+os_x, p[t+1][1]+os1_y), (0,255,0), 3)
            for t in range(17, 20):
                cv2.line(white, (p[t][0]+os_x,   p[t][1]+os1_y),
                                (p[t+1][0]+os_x, p[t+1][1]+os1_y), (0,255,0), 3)
            # Palm connections
            for a, b in [(5,9),(9,13),(13,17),(0,5),(0,17)]:
                cv2.line(white, (p[a][0]+os_x, p[a][1]+os1_y),
                                (p[b][0]+os_x, p[b][1]+os1_y), (0,255,0), 3)
            # Landmark dots
            for i in range(21):
                cv2.circle(white, (p[i][0]+os_x, p[i][1]+os1_y), 2, (0,0,255), 1)

            self.predict_character(white)

            _, buffer = cv2.imencode('.jpg', white)
            skeleton_b64 = ("data:image/jpeg;base64,"
                            + base64.b64encode(buffer).decode('utf-8'))
        except Exception as e:
            print(f"Skeleton draw error: {e}")

        return self._state(skeleton_b64)

    def _state(self, skeleton_b64):
        return {
            "skeleton":    skeleton_b64,
            "character":   self.current_symbol,
            "sentence":    self.str_sentence,
            "suggestions": [self.word1, self.word2, self.word3, self.word4],
            "word":        self.word,
        }

    def predict_character(self, white_img):
        prob = run_inference(white_img)
        ch1  = int(np.argmax(prob))
        prob[ch1] = 0
        ch2 = int(np.argmax(prob))
        prob[ch2] = 0
        ch3 = int(np.argmax(prob))

        pl = [ch1, ch2]
        p  = self.pts

        # ── All gesture classification conditions (unchanged from original) ──

        l = [[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],
             [6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],
             [1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],
             [5,4],[7,0],[7,5],[7,2]]
        if pl in l:
            if (p[6][1]<p[8][1] and p[10][1]<p[12][1] and
                    p[14][1]<p[16][1] and p[18][1]<p[20][1]):
                ch1 = 0

        l = [[2,2],[2,1]]
        if pl in l:
            if p[5][0] < p[4][0]:
                ch1 = 0

        l = [[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[0][0]>p[8][0] and p[0][0]>p[4][0] and p[0][0]>p[12][0]
                    and p[0][0]>p[16][0] and p[0][0]>p[20][0]
                    and p[5][0]>p[4][0]):
                ch1 = 2

        l = [[6,0],[6,6],[6,2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(p[8], p[16]) < 52:
                ch1 = 2

        l = [[1,4],[1,5],[1,6],[1,3],[1,0]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[6][1]>p[8][1] and p[14][1]<p[16][1] and p[18][1]<p[20][1]
                    and p[0][0]<p[8][0] and p[0][0]<p[12][0]
                    and p[0][0]<p[16][0] and p[0][0]<p[20][0]):
                ch1 = 3

        l = [[4,6],[4,1],[4,5],[4,3],[4,7]]
        pl = [ch1, ch2]
        if pl in l:
            if p[4][0] > p[0][0]:
                ch1 = 3

        l = [[5,3],[5,0],[5,7],[5,4],[5,2],[5,1],[5,5]]
        pl = [ch1, ch2]
        if pl in l:
            if p[2][1]+15 < p[16][1]:
                ch1 = 3

        l = [[6,4],[6,1],[6,2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(p[4], p[11]) > 55:
                ch1 = 4

        l = [[1,4],[1,6],[1,1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(p[4],p[11])>50 and p[6][1]>p[8][1]
                    and p[10][1]<p[12][1] and p[14][1]<p[16][1]
                    and p[18][1]<p[20][1]):
                ch1 = 4

        l = [[3,6],[3,4]]
        pl = [ch1, ch2]
        if pl in l:
            if p[4][0] < p[0][0]:
                ch1 = 4

        l = [[2,2],[2,5],[2,4]]
        pl = [ch1, ch2]
        if pl in l:
            if p[1][0] < p[12][0]:
                ch1 = 4

        l = [[3,6],[3,5],[3,4]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[6][1]>p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1]
                    and p[18][1]<p[20][1] and p[4][1]>p[10][1]):
                ch1 = 5

        l = [[3,2],[3,1],[3,6]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[4][1]+17>p[8][1] and p[4][1]+17>p[12][1]
                    and p[4][1]+17>p[16][1] and p[4][1]+17>p[20][1]):
                ch1 = 5

        l = [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]]
        pl = [ch1, ch2]
        if pl in l:
            if p[4][0] > p[0][0]:
                ch1 = 5

        l = [[0,2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[0][0]<p[8][0] and p[0][0]<p[12][0]
                    and p[0][0]<p[16][0] and p[0][0]<p[20][0]):
                ch1 = 5

        l = [[5,7],[5,2],[5,6]]
        pl = [ch1, ch2]
        if pl in l:
            if p[3][0] < p[0][0]:
                ch1 = 7

        l = [[4,6],[4,2],[4,4],[4,1],[4,5],[4,7]]
        pl = [ch1, ch2]
        if pl in l:
            if p[6][1] < p[8][1]:
                ch1 = 7

        l = [[6,7],[0,7],[0,1],[0,0],[6,4],[6,6],[6,5],[6,1]]
        pl = [ch1, ch2]
        if pl in l:
            if p[18][1] > p[20][1]:
                ch1 = 7

        l = [[0,4],[0,2],[0,3],[0,1],[0,6]]
        pl = [ch1, ch2]
        if pl in l:
            if p[5][0] > p[16][0]:
                ch1 = 6

        l = [[7,2]]
        pl = [ch1, ch2]
        if pl in l:
            if p[18][1]<p[20][1] and p[8][1]<p[10][1]:
                ch1 = 6

        l = [[2,1],[2,2],[2,6],[2,7],[2,0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(p[8], p[16]) > 50:
                ch1 = 6

        l = [[4,6],[4,2],[4,1],[4,4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(p[4], p[11]) < 60:
                ch1 = 6

        l = [[1,4],[1,6],[1,0],[1,2]]
        pl = [ch1, ch2]
        if pl in l:
            if p[5][0]-p[4][0]-15 > 0:
                ch1 = 6

        l = [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],
             [6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[6][1]>p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]>p[16][1] and p[18][1]>p[20][1]):
                ch1 = 1

        l = [[6,1],[6,0],[0,3],[6,4],[2,2],[0,6],[6,2],[7,6],[4,6],[4,1],
             [4,2],[0,2],[7,1],[7,4],[6,6],[7,2],[7,5],[7,2]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[6][1]<p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]>p[16][1] and p[18][1]>p[20][1]):
                ch1 = 1

        l = [[6,1],[6,0],[4,2],[4,1],[4,6],[4,4]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[10][1]>p[12][1] and p[14][1]>p[16][1]
                    and p[18][1]>p[20][1]):
                ch1 = 1

        l = [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[6][1]>p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1]
                    and p[18][1]<p[20][1] and p[2][0]<p[0][0]
                    and p[4][1]>p[14][1]):
                ch1 = 1

        l = [[4,1],[4,2],[4,4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(p[4],p[11])<50 and p[6][1]>p[8][1]
                    and p[10][1]<p[12][1] and p[14][1]<p[16][1]
                    and p[18][1]<p[20][1]):
                ch1 = 1

        l = [[3,4],[3,0],[3,1],[3,5],[3,6]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[6][1]>p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1]
                    and p[18][1]<p[20][1] and p[2][0]<p[0][0]
                    and p[14][1]<p[4][1]):
                ch1 = 1

        l = [[6,6],[6,4],[6,1],[6,2]]
        pl = [ch1, ch2]
        if pl in l:
            if p[5][0]-p[4][0]-15 < 0:
                ch1 = 1

        l = [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7,5],[7,1],
             [7,6],[7,7]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[6][1]<p[8][1] and p[10][1]<p[12][1]
                    and p[14][1]<p[16][1] and p[18][1]>p[20][1]):
                ch1 = 1

        l = [[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[4][0]<p[5][0]+15 and p[6][1]<p[8][1]
                    and p[10][1]<p[12][1] and p[14][1]<p[16][1]
                    and p[18][1]>p[20][1]):
                ch1 = 7

        l = [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[6][1]>p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]<p[16][1] and p[18][1]<p[20][1]
                    and p[4][1]>p[14][1]):
                ch1 = 1

        l = [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]
        pl = [ch1, ch2]
        if pl in l:
            fg = 13
            if not (p[0][0]+fg<p[8][0] and p[0][0]+fg<p[12][0]
                    and p[0][0]+fg<p[16][0] and p[0][0]+fg<p[20][0]) \
               and not (p[0][0]>p[8][0] and p[0][0]>p[12][0]
                    and p[0][0]>p[16][0] and p[0][0]>p[20][0]) \
               and self.distance(p[4],p[11])<50:
                ch1 = 1

        l = [[5,0],[5,5],[0,1]]
        pl = [ch1, ch2]
        if pl in l:
            if (p[6][1]>p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]>p[16][1]):
                ch1 = 1

        # ── Subgroup classification ──────────────────────────────────────────
        if ch1 == 0:
            ch1 = 'S'
            if (p[4][0]<p[6][0] and p[4][0]<p[10][0]
                    and p[4][0]<p[14][0] and p[4][0]<p[18][0]):
                ch1 = 'A'
            if (p[4][0]>p[6][0] and p[4][0]<p[10][0] and p[4][0]<p[14][0]
                    and p[4][0]<p[18][0] and p[4][1]<p[14][1]
                    and p[4][1]<p[18][1]):
                ch1 = 'T'
            if (p[4][1]>p[8][1] and p[4][1]>p[12][1]
                    and p[4][1]>p[16][1] and p[4][1]>p[20][1]):
                ch1 = 'E'
            if (p[4][0]>p[6][0] and p[4][0]>p[10][0]
                    and p[4][0]>p[14][0] and p[4][1]<p[18][1]):
                ch1 = 'M'
            if (p[4][0]>p[6][0] and p[4][0]>p[10][0]
                    and p[4][1]<p[18][1] and p[4][1]<p[14][1]):
                ch1 = 'N'

        if ch1 == 2:
            ch1 = 'C' if self.distance(p[12], p[4]) > 42 else 'O'

        if ch1 == 3:
            ch1 = 'G' if self.distance(p[8], p[12]) > 72 else 'H'

        if ch1 == 7:
            ch1 = 'Y' if self.distance(p[8], p[4]) > 42 else 'J'

        if ch1 == 4: ch1 = 'L'
        if ch1 == 6: ch1 = 'X'

        if ch1 == 5:
            if (p[4][0]>p[12][0] and p[4][0]>p[16][0]
                    and p[4][0]>p[20][0]):
                ch1 = 'Z' if p[8][1]<p[5][1] else 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (p[6][1]>p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]>p[16][1] and p[18][1]>p[20][1]):
                ch1 = 'B'
            if (p[6][1]>p[8][1] and p[10][1]<p[12][1]
                    and p[14][1]<p[16][1] and p[18][1]<p[20][1]):
                ch1 = 'D'
            if (p[6][1]<p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]>p[16][1] and p[18][1]>p[20][1]):
                ch1 = 'F'
            if (p[6][1]<p[8][1] and p[10][1]<p[12][1]
                    and p[14][1]<p[16][1] and p[18][1]>p[20][1]):
                ch1 = 'I'
            if (p[6][1]>p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]>p[16][1] and p[18][1]<p[20][1]):
                ch1 = 'W'
            if (p[6][1]>p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]<p[16][1] and p[18][1]<p[20][1]
                    and p[4][1]<p[9][1]):
                ch1 = 'K'
            if ((self.distance(p[8],p[12])-self.distance(p[6],p[10]))<8
                    and p[6][1]>p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]<p[16][1] and p[18][1]<p[20][1]):
                ch1 = 'U'
            if ((self.distance(p[8],p[12])-self.distance(p[6],p[10]))>=8
                    and p[6][1]>p[8][1] and p[10][1]>p[12][1]
                    and p[14][1]<p[16][1] and p[18][1]<p[20][1]
                    and p[4][1]>p[9][1]):
                ch1 = 'V'
            if (p[8][0]>p[12][0] and p[6][1]>p[8][1]
                    and p[10][1]>p[12][1] and p[14][1]<p[16][1]
                    and p[18][1]<p[20][1]):
                ch1 = 'R'

        if ch1 in [1,'E','S','X','Y','B']:
            if (p[6][1]>p[8][1] and p[10][1]<p[12][1]
                    and p[14][1]<p[16][1] and p[18][1]>p[20][1]):
                ch1 = " "

        if ch1 in ['E','Y','B']:
            if (p[4][0]<p[5][0] and p[6][1]>p[8][1]
                    and p[10][1]>p[12][1] and p[14][1]>p[16][1]
                    and p[18][1]>p[20][1]):
                ch1 = "next"

        if ch1 in ['Next','B','C','H','F','X']:
            if (p[0][0]>p[8][0] and p[0][0]>p[12][0]
                    and p[0][0]>p[16][0] and p[0][0]>p[20][0]
                    and p[4][1]<p[8][1] and p[4][1]<p[12][1]
                    and p[4][1]<p[16][1] and p[4][1]<p[20][1]
                    and p[4][1]<p[6][1] and p[4][1]<p[10][1]
                    and p[4][1]<p[14][1] and p[4][1]<p[18][1]):
                ch1 = 'Backspace'

        # ── Sentence building ────────────────────────────────────────────────
        if ch1 == "next" and self.prev_char != "next":
            if self.ten_prev_char[(self.count-2) % 10] != "next":
                prev = self.ten_prev_char[(self.count-2) % 10]
                if prev == "Backspace":
                    self.str_sentence = self.str_sentence[:-1]
                else:
                    self.str_sentence += str(prev)
            else:
                prev = self.ten_prev_char[(self.count-0) % 10]
                if prev != "Backspace":
                    self.str_sentence += str(prev)

        if ch1 == "  " and self.prev_char != "  ":
            self.str_sentence += "  "

        self.prev_char = ch1
        if isinstance(ch1, str) and len(ch1) == 1:
            self.current_symbol = ch1

        self.count += 1
        self.ten_prev_char[self.count % 10] = ch1

        # ── Word suggestions ─────────────────────────────────────────────────
        if len(self.str_sentence.strip()) != 0:
            st = self.str_sentence.rfind(" ")
            word = self.str_sentence[st+1:]
            self.word = word
            if len(word.strip()) != 0:
                safe_word = word[-15:] if len(word) > 15 else word
                suggestions = sorted(ddd.candidates(safe_word) or [])
                lenn = len(suggestions)
                self.word1 = suggestions[0] if lenn >= 1 else " "
                self.word2 = suggestions[1] if lenn >= 2 else " "
                self.word3 = suggestions[2] if lenn >= 3 else " "
                self.word4 = suggestions[3] if lenn >= 4 else " "
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = " "

    def action_suggestion(self, index):
        words = [self.word1, self.word2, self.word3, self.word4]
        selected = words[index]
        if selected and selected.strip():
            idx_space = self.str_sentence.rfind(" ")
            idx_word  = self.str_sentence.find(self.word, idx_space)
            self.str_sentence = self.str_sentence[:idx_word] + selected.upper() + " "
            self.word1 = self.word2 = self.word3 = self.word4 = " "

    def clear(self):
        self.str_sentence = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        self.current_symbol = " "


# ── WebSocket endpoint ───────────────────────────────────────────────────────
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    session = SignLanguageSession()
    try:
        while True:
            data    = await websocket.receive_text()
            message = json.loads(data)

            if message['type'] == 'frame':
                result = await asyncio.get_event_loop().run_in_executor(
                    None, session.process_frame, message['image']
                )
                await websocket.send_json({"type": "prediction", **result})

            elif message['type'] == 'action':
                session.action_suggestion(message['index'])
                await websocket.send_json({
                    "type":        "state_update",
                    "sentence":    session.str_sentence,
                    "suggestions": [session.word1, session.word2,
                                    session.word3, session.word4],
                })

            elif message['type'] == 'clear':
                session.clear()
                await websocket.send_json({
                    "type":        "state_update",
                    "sentence":    session.str_sentence,
                    "suggestions": [session.word1, session.word2,
                                    session.word3, session.word4],
                })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


# Serve landing page last so it doesn't intercept API routes
app.mount("/", StaticFiles(directory="web_portal", html=True), name="web_portal")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
