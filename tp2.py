import cv2
import numpy as np

# === CARGA Y PROCESAMIENTO DE REFERENCIAS ===
ref_images = {
    "corazon": "corazon.png",
    "circulo": "circulo.png",
    "pentagono": "pentagono.png"
}
reference_contours = {}
for label, path in ref_images.items():
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        reference_contours[label] = max(contours, key=cv2.contourArea)

# === PARÁMETROS INICIALES ===
window_name = "Reconocimiento de Formas en Vivo"
thresh_init = 127
max_thresh = 255
morph_size_init = 3
max_morph = 21
area_min = 1000
match_thresh_init = 0.2

def nothing(x): pass

cv2.namedWindow(window_name)
cv2.createTrackbar("Umbral", window_name, thresh_init, max_thresh, nothing)
cv2.createTrackbar("Morph", window_name, morph_size_init, max_morph, nothing)
cv2.createTrackbar("MatchThresh x1000", window_name, int(match_thresh_init*1000), 1000, nothing)

# === CAPTURA DE CÁMARA ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Leer valores de las barras
    thresh_val = cv2.getTrackbarPos("Umbral", window_name)
    morph_size = cv2.getTrackbarPos("Morph", window_name)
    if morph_size % 2 == 0: morph_size += 1
    match_thresh = cv2.getTrackbarPos("MatchThresh x1000", window_name) / 1000.0

    # Procesamiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) > area_min]

    annotated = frame.copy()
    for cnt in filtered:
        best_label = "Desconocido"
        best_score = float("inf")
        for label, ref_cnt in reference_contours.items():
            score = cv2.matchShapes(cnt, ref_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
            if score < match_thresh and score < best_score:
                best_score = score
                best_label = label
        cv2.drawContours(annotated, [cnt], -1, (0,255,0), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(annotated, best_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow(window_name, annotated)
    key = cv2.waitKey(1)
    if key == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()