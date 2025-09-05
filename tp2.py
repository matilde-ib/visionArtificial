import cv2
import numpy as np
import os
from PIL import Image  # <-- Agrega esta importación

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

def cargar_contorno_referencia(nombre):
    base_path = os.path.join(os.path.dirname(__file__), "tp_deteccion", "figures")
    path = os.path.join(base_path, nombre)
    try:
        img_pil = Image.open(path).convert("L")
        img = np.array(img_pil)
    except Exception as e:
        print(f"No se pudo cargar la imagen de referencia: {path} ({e})")
        return None
    # Fondo claro, figura oscura
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print(f"No se encontraron contornos en la imagen de referencia: {nombre}")
        return None
    return max(contornos, key=cv2.contourArea)

def preprocesar(frame, thresh_val, morph_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Fondo claro, figura oscura
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return clean

def main():
    window = "Formas"
    cv2.namedWindow(window)
    cv2.createTrackbar("Umbral", window, 127, 255, lambda x: None)
    cv2.createTrackbar("Morph", window, 5, 31, lambda x: None)
    cv2.createTrackbar("Match x1000", window, 100, 2000, lambda x: None)

    referencias = {
        "corazon": cargar_contorno_referencia("corazon1.png"),
        "circulo": cargar_contorno_referencia("circulo1.png"),
        "pentagono": cargar_contorno_referencia("pentagono1.png")
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        thresh_val = cv2.getTrackbarPos("Umbral", window)
        morph_size = cv2.getTrackbarPos("Morph", window)
        if morph_size < 1: morph_size = 1
        if morph_size % 2 == 0: morph_size += 1
        match_thresh = cv2.getTrackbarPos("Match x1000", window) / 1000.0

        clean = preprocesar(frame, thresh_val, morph_size)
        contornos, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_out = frame.copy()

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < 1000: continue
            mejor_label = "Desconocido"
            mejor_score = float("inf")
            for label, ref in referencias.items():
                if ref is None: continue
                score = cv2.matchShapes(cnt, ref, cv2.CONTOURS_MATCH_I1, 0.0)
                if score < match_thresh and score < mejor_score:
                    mejor_score = score
                    mejor_label = label
            color = COLOR_GREEN if mejor_label == "corazon" else COLOR_RED if mejor_label == "circulo" else COLOR_BLUE if mejor_label == "pentagono" else (255,255,255)
            cv2.drawContours(frame_out, [cnt], -1, color, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(frame_out, mejor_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(window, frame_out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()