# Requisitos: pip install mediapipe opencv-python numpy
import time, random
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# ---------- Utiles para PNG con alfa ----------
def overlay_png_at(bg, fg_bgra, x, y):
    """Pega fg (BGRA) sobre bg (BGR) en (x,y) respetando alfa."""
    if fg_bgra is None:
        return bg
    H, W = bg.shape[:2]
    h, w = fg_bgra.shape[:2]
    if x >= W or y >= H: 
        return bg
    x2, y2 = min(x + w, W), min(y + h, H)
    fg = fg_bgra[0:(y2 - y), 0:(x2 - x)]
    if fg.shape[2] == 3:
        alpha = np.ones((fg.shape[0], fg.shape[1], 1), dtype=np.float32)
        rgb = fg.astype(np.float32)
    else:
        alpha = (fg[:, :, 3:4].astype(np.float32)) / 255.0
        rgb = fg[:, :, :3].astype(np.float32)
    roi = bg[y:y2, x:x2, :].astype(np.float32)
    bg[y:y2, x:x2, :] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return bg

def resized_to_height(img, target_h):
    """Escala manteniendo aspecto para que tenga altura = target_h."""
    if img is None: 
        return None
    h, w = img.shape[:2]
    if h == target_h: 
        return img
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

# ---------- Cargar PNGs (misma carpeta que este .py) ----------
HERE = Path(__file__).resolve().parent
png_V    = cv2.imread(str(HERE / "Peace Emoji.png"),      cv2.IMREAD_UNCHANGED)
png_ILY  = cv2.imread(str(HERE / "Rock Emoji.png"),       cv2.IMREAD_UNCHANGED)
png_THUP = cv2.imread(str(HERE / "Thumbs Up Emoji.png"),  cv2.IMREAD_UNCHANGED)

emoji_png = {"V": png_V, "ILY": png_ILY, "THUP": png_THUP}
for k, img in emoji_png.items():
    if img is None:
        print(f"[WARN] No pude cargar PNG para {k}. Revisá el nombre exacto y la carpeta.")

# ---------- Cámara (macOS) ----------
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # 1 suele ser FaceTime; si no, 0
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# ---------- MediaPipe ----------
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ---------- Juego ----------
TARGETS = [("ILY","ILY"), ("V","V"), ("THUP","THUP")]
ROUND_TIME = 2.0            # tiempo máx para acertar cada objetivo
GAME_TIME  = 10.0           # duración total (lo único que mostramos)
score = 0

current_name, _ = random.choice(TARGETS)
round_start = time.time()
game_start  = time.time()

history = deque(maxlen=5)

# Landmarks
TIP = {"thumb":4, "index":8, "middle":12, "ring":16, "pinky":20}
PIP = {"thumb":3, "index":6, "middle":10, "ring":14, "pinky":18}
MCP = {"thumb":2, "index":5, "middle":9, "ring":13, "pinky":17}

def finger_up(lm, finger, handed):
    tip = lm[TIP[finger]]; pip = lm[PIP[finger]]
    if finger != "thumb":
        return tip.y < pip.y - 0.02
    mcp = lm[MCP["thumb"]]
    if handed == "Right":
        return tip.x < mcp.x - 0.02
    else:
        return tip.x > mcp.x + 0.02

def thumb_up_vertical(lm):
    tip = lm[TIP["thumb"]]; mcp = lm[MCP["thumb"]]
    dx, dy = tip.x - mcp.x, tip.y - mcp.y
    return (abs(dy) > abs(dx)) and (dy < -0.02)

def classify_gesture(hand_lms, handed):
    lm = hand_lms.landmark
    idx = finger_up(lm,"index",handed)
    mid = finger_up(lm,"middle",handed)
    ring= finger_up(lm,"ring",handed)
    pky = finger_up(lm,"pinky",handed)
    th_side = finger_up(lm,"thumb",handed)
    th_up   = thumb_up_vertical(lm)
    if idx and mid and not ring and not pky: return "V"
    if th_up and not idx and not mid and not ring and not pky: return "THUP"
    if idx and pky and not mid and not ring and (th_side or th_up): return "ILY"
    return None

# ---------- Loop ----------
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)

    # detección manos
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)

    detected = None
    hands_found = []
    if res.right_hand_landmarks: hands_found.append(("Right", res.right_hand_landmarks))
    if res.left_hand_landmarks:  hands_found.append(("Left",  res.left_hand_landmarks))

    for handed, hand_lms in hands_found:
        mp_drawing.draw_landmarks(frame, hand_lms, mp_holistic.HAND_CONNECTIONS)
        g = classify_gesture(hand_lms, handed)
        if g: history.append(g)

    if history:
        vals, cnts = np.unique(list(history), return_counts=True)
        detected = vals[np.argmax(cnts)]

    # tiempos
    elapsed_game = time.time() - game_start
    # UI: solo contador del juego y puntaje
    cv2.rectangle(frame, (10,10), (360,110), (0,0,0), -1)
    cv2.putText(frame, f"Juego: {max(0, int(GAME_TIME - elapsed_game))}s",
                (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"Puntaje: {score}",
                (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # dibujar OBJETIVO (PNG) arriba derecha con altura 140 px
    target_png = resized_to_height(emoji_png.get(current_name), 140)
    H, W = frame.shape[:2]
    frame = overlay_png_at(frame, target_png, x=W - (target_png.shape[1] + 20), y=20)

    # lógica del juego: acierto dentro de ROUND_TIME => +1 y nuevo objetivo
    if detected == current_name and (time.time() - round_start) <= ROUND_TIME:
        score += 1
        cv2.putText(frame, "¡ACIERTO!", (W//2 - 120, H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
        current_name, _ = random.choice(TARGETS)
        round_start = time.time()
        history.clear()
    # si se acaba el tiempo de la ronda y no acertaste, solo cambia objetivo (sin mensaje)
    elif (time.time() - round_start) > ROUND_TIME:
        current_name, _ = random.choice(TARGETS)
        round_start = time.time()
        history.clear()

    cv2.imshow("Juego de rapidez", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if elapsed_game >= GAME_TIME: break

# ---------- Fin ----------
holistic.close()
cap.release()

final = np.zeros((320, 720, 3), dtype=np.uint8)
cv2.putText(final, "Juego terminado", (150,130),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
cv2.putText(final, f"Puntaje final: {score}", (170,200),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
cv2.imshow("Resultado", final)
cv2.waitKey(3000)
cv2.destroyAllWindows()
