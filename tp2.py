import cv2
import numpy as np
import os

# === UTILIDADES DEL PROYECTO ===
from tp_deteccion.contour import get_contours, filter_contours_by_area
from tp_deteccion.frame_editor import threshold, draw_contours
from tp_deteccion.trackbar import create_trackbar, get_trackbar_value

# === RUTAS DE IMÁGENES DE REFERENCIA ===
# Se asume que las imágenes están en la carpeta actual o especifica la ruta completa
ref_images = {
    "corazon": "corazon.png",
    "circulo": "circulo.png",
    "pentagono": "pentagono.png"
}

# === PARÁMETROS INICIALES ===
window_name = "Reconocimiento de Formas"
thresh_init = 127
max_thresh = 255
morph_size_init = 3
max_morph = 21
area_min = 1000  # área mínima para filtrar contornos espúreos
match_thresh_init = 0.2  # umbral de matchShapes

# === CARGA Y PROCESAMIENTO DE REFERENCIAS ===
reference_contours = {}
for label, path in ref_images.items():
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Binarización simple para referencia
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Encontrar contornos
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Guardar el contorno más grande como referencia
    if contours:
        reference_contours[label] = max(contours, key=cv2.contourArea)

# === FUNCIÓN DE PROCESAMIENTO PRINCIPAL ===
def process_image(img_path, thresh_val, morph_size, match_thresh):
    # 1. Leer imagen y convertir a escala de grises
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Aplicar threshold (binarización)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # 3. Operaciones morfológicas para eliminar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 4. Encontrar todos los contornos
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filtrar contornos pequeños
    filtered = [c for c in contours if cv2.contourArea(c) > area_min]

    # 6. Clasificar cada contorno
    annotated = img.copy()
    for cnt in filtered:
        best_label = "Desconocido"
        best_score = float("inf")
        # Comparar con cada referencia usando matchShapes
        for label, ref_cnt in reference_contours.items():
            score = cv2.matchShapes(cnt, ref_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
            if score < match_thresh and score < best_score:
                best_score = score
                best_label = label
        # Dibujar contorno y etiqueta
        cv2.drawContours(annotated, [cnt], -1, (0,255,0), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(annotated, best_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return annotated

# === INTERFAZ CON BARRAS DE DESPLAZAMIENTO ===
def nothing(x): pass

cv2.namedWindow(window_name)
cv2.createTrackbar("Umbral", window_name, thresh_init, max_thresh, nothing)
cv2.createTrackbar("Morph", window_name, morph_size_init, max_morph, nothing)
cv2.createTrackbar("MatchThresh x1000", window_name, int(match_thresh_init*1000), 1000, nothing)

# === PROCESAR TODAS LAS IMÁGENES DE TEST ===
test_images = ["corazon.png", "circulo.png", "pentagono.png"]
current_idx = 0

while True:
    # Leer valores de las barras
    thresh_val = cv2.getTrackbarPos("Umbral", window_name)
    morph_size = cv2.getTrackbarPos("Morph", window_name)
    if morph_size % 2 == 0: morph_size += 1  # El kernel debe ser impar
    match_thresh = cv2.getTrackbarPos("MatchThresh x1000", window_name) / 1000.0

    # Procesar imagen actual
    img_path = test_images[current_idx]
    annotated = process_image(img_path, thresh_val, morph_size, match_thresh)

    # Mostrar resultado
    cv2.imshow(window_name, annotated)
    key = cv2.waitKey(100)
    if key == 27:  # ESC para salir
        break
    elif key == ord('n'):  # Siguiente imagen
        current_idx = (current_idx + 1) % len(test_images)
    elif key == ord('p'):  # Imagen anterior
        current_idx = (current_idx - 1) % len(test_images)

cv2.destroyAllWindows()