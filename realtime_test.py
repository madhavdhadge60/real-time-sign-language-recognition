import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import time
from collections import deque, Counter
import pyttsx3
import threading

# ================= AUTOCOMPLETE =================
WORDS = ["HELLO", "COME", "SUCCESS", "THANK", "PLEASE", "YES", "NO",
         "HOW", "ARE", "YOU", "HERE", "GIVE", "WATER",
         "NEED", "NICE", "GREAT", "OKAY", "DONE"]

def autocomplete(text):
    words = text.split(" ")
    last_word = words[-1]

    for word in WORDS:
        if word.startswith(last_word) and last_word != "":
            return word

    return ""

# ================= SELF ATTENTION =================
from tensorflow.keras.layers import Layer

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.Wq = self.add_weight(shape=(self.dim, self.dim), initializer='glorot_uniform')
        self.Wk = self.add_weight(shape=(self.dim, self.dim), initializer='glorot_uniform')
        self.Wv = self.add_weight(shape=(self.dim, self.dim), initializer='glorot_uniform')

    def call(self, x):
        Q = tf.matmul(x, self.Wq)
        K = tf.matmul(x, self.Wk)
        V = tf.matmul(x, self.Wv)
        scores = tf.matmul(Q, K, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.dim, tf.float32))
        attn = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(attn, V)

# ================= CONFIG =================
MODEL_PATH = "MD_model_1776730680.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = 160

CONF_THRESHOLD = 0.70
BUFFER_SIZE = 15
STABLE_FRAMES = 10

# ================= LOAD =================
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'SelfAttention': SelfAttention}
)

with open(LABELS_PATH) as f:
    labels = json.load(f)

idx_to_class = {int(k): v for k, v in labels.items()}

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

# ================= TEXT SYSTEM =================
sentence = ""
last_char = ""
last_time = time.time()

# ✅ NEW VARIABLES
DOUBLE_DELAY = 2.5       # repeat delay
cooldown_time = 2.0      # global cooldown
last_added_time = 0

# ================= VOICE =================
def speak_async(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    threading.Thread(target=run).start()

# ================= BUFFERS =================
prob_buffer = deque(maxlen=BUFFER_SIZE)
pred_buffer = deque(maxlen=BUFFER_SIZE)

stable_sign = ""

# ================= FUNCTIONS =================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5,5), 0)
    return img.astype(np.float32) / 255.0

def detect_hand(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        h, w = frame.shape[:2]
        lm = res.multi_hand_landmarks[0].landmark

        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]

        x1, x2 = max(0, min(xs)-20), min(w, max(xs)+20)
        y1, y2 = max(0, min(ys)-20), min(h, max(ys)+20)

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    return None, None

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
prev_time = time.time()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    crop, bbox = detect_hand(frame)

    if crop is None:
        cv2.putText(display, "No hand detected", (10,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        pred_buffer.clear()
        stable_sign = ""

    else:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)

        img = preprocess(crop)
        img = np.expand_dims([img], axis=0)

        probs = model.predict(img, verbose=0)[0]
        prob_buffer.append(probs)

        avg_probs = np.mean(prob_buffer, axis=0)
        idx = np.argmax(avg_probs)
        conf = avg_probs[idx]
        current = idx_to_class[idx]

        cv2.putText(display, f"Conf: {conf:.2f}", (10,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if conf > CONF_THRESHOLD else (0,0,255), 2)

        if conf > CONF_THRESHOLD:
            pred_buffer.append(current)

        if len(pred_buffer) >= STABLE_FRAMES:
            stable_sign = Counter(pred_buffer).most_common(1)[0][0]

    # ================= UPDATED TEXT LOGIC =================
    if stable_sign != "":
        current_time = time.time()

        # GLOBAL COOLDOWN
        if current_time - last_added_time >= cooldown_time:

            # FULL DELETE
            if stable_sign == "del":
                sentence = ""
                last_char = ""
                last_added_time = current_time

            # SPACE
            elif stable_sign == "space":
                sentence += " "
                last_char = stable_sign
                last_added_time = current_time

            # NORMAL LETTER
            else:
                if stable_sign != last_char or (current_time - last_time > DOUBLE_DELAY):
                    sentence += stable_sign
                    last_char = stable_sign
                    last_time = current_time
                    last_added_time = current_time

    suggestion = autocomplete(sentence)

    # ===== FPS =====
    curr = time.time()
    fps = int(1/(curr - prev_time + 1e-6))
    prev_time = curr

    # ===== UI =====
    display_sign = stable_sign if stable_sign != "" else "Waiting..."
    h, w = display.shape[:2]

    cv2.rectangle(display, (0,0), (w,80), (40,40,40), -1)

    title = "SIGN LANGUAGE SYSTEM"
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

    cv2.putText(display, title,
                ((w - text_size[0]) // 2, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(display, f"Sign: {display_sign}", (10,110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.putText(display, f"Next: {suggestion}", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

    cv2.putText(display, f"FPS: {fps}", (w-120, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cursor = "|" if int(time.time()*2)%2==0 else ""

    cv2.putText(display, f"Text: {sentence}{cursor}", (12,162),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)

    cv2.putText(display, f"Text: {sentence}{cursor}", (10,160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

    cv2.imshow("FINAL Sign System", display)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        if sentence:
            sentence = sentence[:-1]
            last_char = ""
    elif key & 0xFF == ord('w'):
        sentence = sentence.rstrip()
        sentence = " ".join(sentence.split(" ")[:-1])
    elif key & 0xFF == ord('v'):
        speak_async(sentence)
    elif key & 0xFF == ord('s'):
        if suggestion != "":
            words = sentence.split(" ")

            # replace only last word
            words[-1] = suggestion

            sentence = " ".join(words) + " "   # add space after word

cap.release()
cv2.destroyAllWindows()