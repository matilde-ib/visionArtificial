# run_classifier.py — Clasificación en tiempo real (modelo.joblib)
import cv2, numpy as np, math
from joblib import load
from pathlib import Path

# --- Rutas ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "modelo.joblib"
print("[MODEL IN]", MODEL_PATH)

# --- Paleta por clase ---
COLOR = {"corazon": (0,255,0), "circulo": (0,0,255), "pentagono": (255,0,0)}

# --- Features (igual que en el generador) ---
def hu_log_sign(cnt):
    m  = cv2.moments(cnt)
    hu = cv2.HuMoments(m).flatten()
    for i in range(7):
        if hu[i] != 0:
            hu[i] = -1 * math.copysign(1.0, hu[i]) * math.log10(abs(hu[i]))
    return hu

def hsv_means_inside_contour(frame_bgr, cnt):
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    H, S, V, _ = cv2.mean(hsv, mask=mask)
    return H, S, V

def preprocess_hsv_dark(frame, morph_k, vmax=110):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V    = hsv[:, :, 2]
    mask = cv2.inRange(V, 0, int(vmax))
    k = max(1, int(morph_k) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    clean  = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return clean

def pick_best_contour(frame, contours, amin, amax):
    H, W = frame.shape[:2]
    frame_area = H * W
    cands = [c for c in contours
             if amin <= cv2.contourArea(c) <= min(amax, 0.8 * frame_area)]
    if not cands: return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V   = hsv[:, :, 2]
    def mean_V_inside(c):
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(m, [c], -1, 255, thickness=cv2.FILLED)
        return cv2.mean(V, mask=m)[0]
    return min(cands, key=mean_V_inside)

def main():
    clf = load(MODEL_PATH)

    window = "Window"
    cv2.namedWindow(window)
    cv2.createTrackbar("Threshold",      window, 110, 255, lambda x: None)   # Vmax(oscuro)
    cv2.createTrackbar("Kernel denoise", window, 7,   31,  lambda x: None)
    cv2.createTrackbar("Min Area",       window, 3000,200000,lambda x: None)
    cv2.createTrackbar("Max Area",       window,120000,400000,lambda x: None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ok, frame = cap.read()
        if not ok: break

        vmax = cv2.getTrackbarPos("Threshold",      window)
        k    = cv2.getTrackbarPos("Kernel denoise", window)
        amin = cv2.getTrackbarPos("Min Area",       window)
        amax = cv2.getTrackbarPos("Max Area",       window)

        mask = preprocess_hsv_dark(frame, k, vmax=vmax)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = pick_best_contour(frame, contours, amin, amax)

        out = frame.copy()
        if cnt is not None:
            hu = hu_log_sign(cnt)
            Hm, Sm, Vm = hsv_means_inside_contour(frame, cnt)
            sample = np.array(list(hu) + [Hm, Sm, Vm], dtype=np.float32).reshape(1, -1)

            label = clf.predict(sample)[0]           # 'corazon'/'circulo'/'pentagono'
            color = COLOR.get(label, (255,255,255))

            cv2.drawContours(out, [cnt], -1, color, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(out, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(window, out)
        cv2.imshow("mask_debug", mask)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
