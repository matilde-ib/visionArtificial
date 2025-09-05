## Guia de deteccion
# Script de ejemplo para clasificar una imagen 2D con Hu Moments

import cv2
import numpy as np
from tp_deteccion.machine.utils.hu_moments_generation import hu_moments_of_file
from tp_deteccion.ml_model import train_model, int_to_label

# 1. Entrenar o cargar el modelo
model = train_model()  # Usa el CSV generado previamente

# 2. Extraer los Hu Moments de la imagen a clasificar
ruta_imagen = "RUTA/A/TU/IMAGEN.png"
hu = hu_moments_of_file(ruta_imagen)  # Devuelve un array (7,1)

# 3. Preparar el vector para el modelo
sample = np.array(hu, dtype=np.float32).reshape(1, -1)

# 4. Predecir la clase
pred = model.predict(sample)[0]
label = int_to_label(pred)

print(f"La imagen {ruta_imagen} fue clasificada como: {label}")