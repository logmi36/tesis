import os
import cv2
from ultralytics import YOLO
import datetime

import sqlite3

conn = sqlite3.connect("C:\data\media\camera.sqlite")
cursor = conn.cursor()


def write(cad):
    dt=datetime.datetime.now()
    dtt=dt.strftime("%H%M%S")
    print('\n')
    print('='*80)
    print(dtt,'\t',cad,' ======')
    print('\n')


def show(list):
    dt=datetime.datetime.now()
    dtt=dt.strftime("%H%M%S")
    print(dtt,*list, sep='\t')
    

# Ruta al modelo YOLO entrenado
MODEL_PATH = r"C:\data\media\wrk\entrenamiento_yolo\exp_yolo\weights\best.pt"

# Carpeta con imágenes originales de vehículos
IMAGES_DIR = r"C:\data\media\dcim"

# Carpeta de salida para placas recortadas
OUTPUT_DIR = "C:\data\media\wrk\plates"

# Umbral minimo de confianza
CONF_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# cargar modelo
model = YOLO(MODEL_PATH)

write('PROCESAMIENTO DE IMÁGENES')

for image_name in os.listdir(IMAGES_DIR):

    image_path = os.path.join(IMAGES_DIR, image_name)

    if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo leer: {image_name}")
        continue

    results = model(image, conf=CONF_THRESHOLD)

    for idx, box in enumerate(results[0].boxes):

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        placa_recortada = image[y1:y2, x1:x2]

        if placa_recortada.size == 0:
            continue

        output_name = f"{os.path.splitext(image_name)[0]}_{idx+1}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        cv2.imwrite(output_path, placa_recortada)

    show([image_name])

write("✅ Recorte de placas finalizado.")
