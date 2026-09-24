import os
import torch
from ultralytics import YOLO
import datetime


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
    


write("1. VERIFICAR GPU")

print("PyTorch version:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU detectada:", torch.cuda.get_device_name(0))
else:
    print("⚠ Entrenando en CPU (MUY lento)")


# ---------------------------------------------------
# 2. RUTAS DEL MODELO Y DATASET
# ---------------------------------------------------
DATASET_YAML = "data.yaml"      # Archivo YAML con rutas al dataset YOLO
BASE_MODEL = "yolov8n.pt"          # Puedes cambiar a yolov8s.pt o yolov8m.pt
OUTPUT_DIR = "entrenamiento_yolo"


# ---------------------------------------------------
# 3. CARGA DEL MODELO
# ---------------------------------------------------
write("📌 Cargando modelo base...")
model = YOLO(BASE_MODEL)


# ---------------------------------------------------
# 4. PARAMETROS DE ENTRENAMIENTO (OPTIMIZADOS)
# ---------------------------------------------------
train_params = dict(
    data=DATASET_YAML,
    imgsz=640,                  # tamaño de imágenes
    epochs=150,                 # modificar según tu tiempo y hardware
    batch=16,                   # aumentar si tienes más VRAM
    workers=4,                  # threads para cargar datos
    device="cpu",                   # GPU 0

    # Optimizaciones recomendadas
    lr0=0.001,                  # learning rate inicial
    lrf=0.01,                   # factor final de LR
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=0.20,                # Mosaic bajo para no distorsionar placas
    mixup=0.0,                  # NO recomendado para placas → lo desactivamos

    # Regularización
    dropout=0.05,

    # Aumentos seguros para placas
    degrees=0,
    translate=0.1,
    scale=0.5,
    shear=0,

    # Guardado de pesos
    project=OUTPUT_DIR,
    name="exp_yolo",
    exist_ok=True
)


# ---------------------------------------------------
# '5. INICIAR ENTRENAMIENTO'
# ---------------------------------------------------

write("🚀 Iniciando entrenamiento YOLO...")

results = model.train(**train_params)

write("🎉 ENTRENAMIENTO COMPLETADO")
write("Los resultados se guardaron en:")
print(results.save_dir)

write("eof")
