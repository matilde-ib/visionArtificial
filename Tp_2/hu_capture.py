# hu_capture.py — Generador de descriptores (Hu + color HSV)
import cv2, numpy as np, math, csv, os
from pathlib import Path

# --- Rutas (se guarda todo dentro de Tp_2/) ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "machine" / "generated-files"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH  = DATA_DIR / "shapes-hu-moments.csv"
print("[SAVE_PATH]", SAVE_PATH)

# --- Etiquetas disponibles (teclas 1/2/3) ---
LABELS = {1: "corazon", 2: "circulo", 3: "pentagono"}
current_label = 1  # arranca en corazón

# --- Utilidades CSV (para undo/relabel rápidos con Z/X) ---
def _csv_read_rows(path: Path):
    if not path.exists(): return []
    with path.open(newline="") as f:
        return list(csv.reader(f))

def _csv_write_rows(path: Path, rows):
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

def undo_last_row(path: Path):
    rows = _csv_read_rows(path)
    if len(rows) <= 1:
        print("[UNDO] No hay filas para borrar.")
        return
    borrada = rows.pop()
    _csv_write_rows(path, rows)
    print("[UNDO] Eliminada última fila:", borrada)

def relabel_last_row(path: Path, new_label: str):
    rows = _csv_read_rows(path)
    if len(rows) <= 1:
        print("[RELAB] No hay filas para editar.")
        return
    rows[-1][-1] = new_label
    _csv_write_rows(path, rows)
    print(f"[RELAB] Última fila cambiada a '{new_label}'")

# --- Cálculo de features ---
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
    """Segmenta regiones oscuras: V <= vmax (HSV) y limpia con morfología."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V    = hsv[:, :, 2]
    mask = cv2.inRange(V, 0, int(vmax))
    k = max(1, int(morph_k) | 1)  # kernel impar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    clean  = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return clean

def pick_best_contour(frame, contours, amin, amax):
    """Filtra por área [amin, amax] y por debajo del 80% del frame.
       Devuelve el contorno más oscuro (menor V promedio)."""
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
    return min(cands, key=mean_V_inside)  # menor V => más oscuro

def main():
    global current_label

    # UI similar al profe
    window = "Window"
    cv2.namedWindow(window)
    cv2.createTrackbar("Threshold",      window, 110, 255, lambda x: None)   # Vmax(oscuro)
    cv2.createTrackbar("Kernel denoise", window, 7,   31,  lambda x: None)   # tamaño kernel
    cv2.createTrackbar("Min Area",       window, 3000,200000,lambda x: None)
    cv2.createTrackbar("Max Area",       window,120000,400000,lambda x: None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    saved = 0
    print(f"[INFO] Clase actual: {LABELS[current_label]} (teclas 1/2/3)")

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
            cv2.drawContours(out, [cnt], -1, (0,255,0), 2)

        cv2.putText(out, f"Clase: {LABELS[current_label]} (1/2/3)", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(out, f"[ESPACIO] guarda Hu+HSV | Guardadas: {saved}", (10,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(window, out)
        cv2.imshow("mask_debug", mask)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('1'): current_label = 1; print("[INFO] -> corazon")
        elif key == ord('2'): current_label = 2; print("[INFO] -> circulo")
        elif key == ord('3'): current_label = 3; print("[INFO] -> pentagono")
        elif key == ord('z'): undo_last_row(SAVE_PATH)
        elif key == ord('x'): relabel_last_row(SAVE_PATH, LABELS[current_label])
        elif key in [27, ord('q')]: break
        elif key == ord(' '):
            if cnt is None:
                print("[WARN] Sin contorno válido."); continue
            hu = hu_log_sign(cnt)
            Hm, Sm, Vm = hsv_means_inside_contour(frame, cnt)
            row = list(map(float, hu)) + [float(Hm), float(Sm), float(Vm), LABELS[current_label]]

            write_header = not SAVE_PATH.exists()
            with SAVE_PATH.open("a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow([f"hu{i}" for i in range(1,8)] + ["H_mean","S_mean","V_mean","label"])
                w.writerow(row)
            saved += 1
            print(f"[OK] #{saved} -> label={LABELS[current_label]} | Hu={list(hu)} HSV=({Hm:.1f},{Sm:.1f},{Vm:.1f})")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
