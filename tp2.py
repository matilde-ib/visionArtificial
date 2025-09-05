import cv2
import numpy as np
import math

# Importa funciones de tu proyecto
from tp_deteccion.contour import get_contours, filter_contours_by_area, get_bounding_rect
from tp_deteccion.frame_editor import apply_color_convertion, threshold, denoise, draw_contours
from tp_deteccion.trackbar import create_trackbar, get_trackbar_value

# Colores para anotaciones
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

# Carga los contornos de referencia de las imágenes de entrenamiento
def get_reference_contours():
    refs = {}
    for label, path in [("corazon", "corazon.png"), ("circulo", "circulo.png"), ("pentagono", "pentagono.png")]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours = get_contours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            refs[label] = max(contours, key=cv2.contourArea)
    return refs

def main():
    window_name = 'Reconocimiento de Formas'
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(0)

    # Barras de ajuste
    create_trackbar('Umbral', window_name, 255)
    create_trackbar('Morph', window_name, 21)
    create_trackbar('MatchThresh x1000', window_name, 1000)

    reference_contours = get_reference_contours()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Leer valores de las barras
        thresh_val = get_trackbar_value('Umbral', window_name)
        morph_size = get_trackbar_value('Morph', window_name)
        if morph_size < 1: morph_size = 1
        if morph_size % 2 == 0: morph_size += 1
        match_thresh = get_trackbar_value('MatchThresh x1000', window_name) / 1000.0

        # Procesamiento
        gray = apply_color_convertion(frame, cv2.COLOR_BGR2GRAY)
        binary = threshold(gray, 255, cv2.THRESH_BINARY, thresh_val)
        clean = denoise(binary, cv2.MORPH_ELLIPSE, morph_size)
        contours = get_contours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = filter_contours_by_area(contours, 1000, 1e6)

        annotated = frame.copy()
        for cnt in filtered:
            best_label = "Desconocido"
            best_score = float("inf")
            for label, ref_cnt in reference_contours.items():
                score = cv2.matchShapes(cnt, ref_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                if score < match_thresh and score < best_score:
                    best_score = score
                    best_label = label
            color = COLOR_GREEN if best_label == "corazon" else COLOR_RED if best_label == "circulo" else COLOR_BLUE if best_label == "pentagono" else (255,255,255)
            draw_contours(annotated, [cnt], color, 2)
            x, y, _, _ = get_bounding_rect(cnt)
            cv2.putText(annotated, best_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()