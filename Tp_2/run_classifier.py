# run_classifier.py — Solo Hu, Otsu/Manual Threshold con sliders
import cv2, numpy as np, math
from joblib import load
from pathlib import Path

# --- Rutas ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "modelo.joblib"
print("[MODEL IN]", MODEL_PATH)

# --- Colores por clase (para dibujar) ---
COLOR = {"corazon": (0,255,0), "circulo": (0,0,255), "pentagono": (255,0,0)}

# --- Descriptor Hu (log-sign) ---
def hu_log_sign(cnt):
    m  = cv2.moments(cnt)
    hu = cv2.HuMoments(m).flatten()
    for i in range(7):
        if hu[i] != 0:
            hu[i] = -1 * math.copysign(1.0, hu[i]) * math.log10(abs(hu[i]))
    return hu

# --- Otsu (normal e invertido) ---
def masks_otsu(frame, morph_k):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, b1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, b2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(1, int(morph_k) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    b1 = cv2.morphologyEx(b1, cv2.MORPH_OPEN,  kernel)
    b1 = cv2.morphologyEx(b1, cv2.MORPH_CLOSE, kernel)
    b2 = cv2.morphologyEx(b2, cv2.MORPH_OPEN,  kernel)
    b2 = cv2.morphologyEx(b2, cv2.MORPH_CLOSE, kernel)
    return b1, b2

# --- Manual (usa slider de Threshold, normal e invertido) ---
def masks_manual(frame, thr, morph_k):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, b1 = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    _, b2 = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    k = max(1, int(morph_k) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    b1 = cv2.morphologyEx(b1, cv2.MORPH_OPEN,  kernel)
    b1 = cv2.morphologyEx(b1, cv2.MORPH_CLOSE, kernel)
    b2 = cv2.morphologyEx(b2, cv2.MORPH_OPEN,  kernel)
    b2 = cv2.morphologyEx(b2, cv2.MORPH_CLOSE, kernel)
    return b1, b2

# --- Elegir mejor contorno de una máscara ---
def pick_best_contour(frame, mask, amin, amax):
    H, W = frame.shape[:2]
    frame_area = H * W
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = [c for c in contours
             if amin <= cv2.contourArea(c) <= min(amax, 0.8 * frame_area)]
    if not cands:
        return None
    return max(cands, key=cv2.contourArea)

def main():
    clf = load(MODEL_PATH)

    # UI (al estilo del profe)
    window = "Window"
    cv2.namedWindow(window)
    cv2.createTrackbar("Threshold",       window, 127, 255, lambda x: None)  # manual
    cv2.createTrackbar("Use Otsu (0/1)",  window, 1,   1,   lambda x: None)  # 1=Otsu, 0=Manual
    cv2.createTrackbar("Kernel denoise",  window, 7,   31,  lambda x: None)
    cv2.createTrackbar("Min Area",        window, 3000,200000, lambda x: None)
    cv2.createTrackbar("Max Area",        window,120000,400000, lambda x: None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ok, frame = cap.read()
        if not ok: break

        thr   = cv2.getTrackbarPos("Threshold",      window)
        use_o = cv2.getTrackbarPos("Use Otsu (0/1)", window)  # 1=Otsu
        k     = cv2.getTrackbarPos("Kernel denoise", window)
        amin  = cv2.getTrackbarPos("Min Area",       window)
        amax  = cv2.getTrackbarPos("Max Area",       window)

        # Dos máscaras (normal e invertida), por Otsu o Manual
        m1, m2 = (masks_otsu(frame, k) if use_o == 1 else masks_manual(frame, thr, k))
        c1 = pick_best_contour(frame, m1, amin, amax)
        c2 = pick_best_contour(frame, m2, amin, amax)

        # Elegir el mejor contorno de ambos (por área)
        candidates = [c for c in (c1, c2) if c is not None]
        cnt = max(candidates, key=cv2.contourArea) if candidates else None

        # Para debug, mostramos la máscara “ganadora”
        mask_debug = m1 if cnt is c1 else (m2 if cnt is c2 else m1)

        out = frame.copy()
        if cnt is not None:
            hu = hu_log_sign(cnt)
            sample = np.array(list(hu), dtype=np.float32).reshape(1, -1)
            label = clf.predict(sample)[0]  # 'corazon'/'circulo'/'pentagono'
            color = COLOR.get(label, (255,255,255))

            cv2.drawContours(out, [cnt], -1, color, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(out, str(label), (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(window, out)
        cv2.imshow("mask_debug", mask_debug)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
