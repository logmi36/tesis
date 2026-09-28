"""
Script: train_trocr_v2.py
Descripcion: Fine-tuning de TrOCR (v2) para reconocimiento de placas
             vehiculares peruanas.

Motivo del nuevo entrenamiento
------------------------------
El modelo anterior (trocr_model_final, config "eval3" de den32.py) se
evaluo sobre el split de TEST y tambien sobre el split de VALIDATION
usando imagenes crudas identicas a las de entrenamiento (para descartar
un problema de preprocesamiento). En ambos casos el resultado fue casi
identico: 0% Exact Match, CER ~0.43, con un patron sistematico de
caracteres omitidos y colapso por repeticion (ej. "F7R701" ->
"FRRRRRRRRR701", "Y2B362" -> "YBBBBBBB362"). Esto descarta un bug de
preprocesamiento y apunta a un modelo genuinamente subentrenado.

Ajustes concretos respecto al entrenamiento anterior
-----------------------------------------------------
1. EarlyStoppingCallback real sobre el split de VALIDACION oficial de
   la tabla `plates` (el script anterior hacia un split aleatorio
   adicional dentro del propio train, en vez de usar la particion de
   validacion ya definida en la base de datos).
2. Penalizacion de repeticion en la generacion (no_repeat_ngram_size,
   repetition_penalty) para mitigar el colapso por repeticion observado.
3. Carga de imagenes perezosa (Dataset con __getitem__), evitando cargar
   miles de imagenes en RAM de una sola vez.
4. Split train/validation/test tomado directamente de la tabla `plates`
   (kd='train'/'validation'/'test') - el set de test NUNCA se toca aqui.
5. Soporte de aumento de datos (rotacion leve, color jitter, blur) ya
   implementado (AUGMENT_N), pero desactivado por defecto (AUGMENT_N=0)
   dado que este hardware es CPU-only y el tiempo de entrenamiento por
   epoca escala linealmente con el tamano del dataset (~4h/epoca sin
   augmentar vs ~12h/epoca con augmentacion x3 sobre las 2500 imagenes
   de train). Activar AUGMENT_N>0 si se dispone de mas tiempo/hardware.

Autor: Irwin Ivan Mendoza Gonzalez
Tesis: UNI 2025
"""

import os

os.environ["USE_TF"] = "0"  # evitar que transformers intente cargar TF/Keras

import random
import sqlite3
import datetime

import pandas as pd
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

os.environ["WANDB_DISABLED"] = "true"

# =============================================================================
# CONFIGURACION
# =============================================================================

DB_PATH    = r"C:\Tesis2\camera.sqlite"
# plates_mod es la carpeta ORIGINAL de entrenamiento (equivalente a la
# ruta "C:\Users\pc\Documents\camera\plates_mod" de den32.py, ya
# recuperada localmente): imagenes de placa ya cuadradas a 384x384,
# el mismo formato usado en el entrenamiento original. Cubre el 100%
# de los archivos de la tabla `plates`.
IMG_DIRS   = [r"C:\data\media\plates_mod"]

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT      = os.path.join(BASE_DIR, "trocr_model_final_v2")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR        = os.path.join(BASE_DIR, "logs")

MODELO_BASE    = "microsoft/trocr-base-printed"

AUGMENT_N      = 0      # variantes sinteticas por imagen de train (0 = sin augmentar)
EPOCHS         = 8
LEARNING_RATE  = 3e-5
BATCH_SIZE     = 8
MAX_LABEL_LEN  = 16
GEN_MAX_LENGTH = 16
EARLY_STOP_PATIENCE = 3
SEED           = 42

random.seed(SEED)
torch.manual_seed(SEED)


def write(cad):
    dt = datetime.datetime.now().strftime("%H%M%S")
    print("\n" + "=" * 80)
    print(dt, "\t", cad, " ======")
    print()


def buscar_imagen(nombre):
    for d in IMG_DIRS:
        ruta = os.path.join(d, nombre)
        if os.path.exists(ruta):
            return ruta
    return None


# =============================================================================
# AUMENTO DE DATOS (solo train)
# =============================================================================

_color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)


def augmentar_imagen(img):
    """Genera una variante de la placa: rotacion leve, color jitter y blur leve."""
    angulo = random.uniform(-4, 4)
    aug = img.rotate(angulo, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    aug = _color_jitter(aug)
    if random.random() < 0.4:
        aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    return aug


# =============================================================================
# DATASET (carga perezosa)
# =============================================================================

class PlacasDataset(torch.utils.data.Dataset):
    """
    registros: lista de tuplas (ruta_imagen, texto).
    Si augment=True, cada registro se expande a (1 + n_aug) entradas:
    la original y n_aug variantes aumentadas, generadas al vuelo en
    __getitem__ (no se guardan en memoria).
    """

    def __init__(self, registros, processor, max_label_len, augment=False, n_aug=0):
        self.processor = processor
        self.max_label_len = max_label_len
        self.items = []
        for ruta, texto in registros:
            self.items.append((ruta, texto, False))
            if augment:
                for _ in range(n_aug):
                    self.items.append((ruta, texto, True))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ruta, texto, aplicar_aug = self.items[idx]
        imagen = Image.open(ruta).convert("RGB")
        if aplicar_aug:
            imagen = augmentar_imagen(imagen)

        pixel_values = self.processor(images=imagen, return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(
            texto, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_label_len,
        ).input_ids[0]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    pixel_values = torch.stack([ej["pixel_values"] for ej in batch])
    labels = pad_sequence([ej["labels"] for ej in batch], batch_first=True, padding_value=-100)
    return {"pixel_values": pixel_values, "labels": labels}


# =============================================================================
# CARGA DE PARTICIONES
# =============================================================================

write("Cargando particiones desde la base de datos")
conn = sqlite3.connect(DB_PATH)
df_train = pd.read_sql("select txt2 as archivo, rem as texto from plates where kd='train'", conn)
df_val = pd.read_sql("select txt2 as archivo, rem as texto from plates where kd='validation'", conn)
conn.close()

registros_train, faltantes_train = [], 0
for _, row in df_train.iterrows():
    ruta = buscar_imagen(row["archivo"])
    if ruta is None:
        faltantes_train += 1
        continue
    registros_train.append((ruta, str(row["texto"]).strip().upper()))

registros_val, faltantes_val = [], 0
for _, row in df_val.iterrows():
    ruta = buscar_imagen(row["archivo"])
    if ruta is None:
        faltantes_val += 1
        continue
    registros_val.append((ruta, str(row["texto"]).strip().upper()))

print(f"  Train: {len(registros_train)} imagenes encontradas ({faltantes_train} faltantes)")
print(f"  Validation: {len(registros_val)} imagenes encontradas ({faltantes_val} faltantes)")

write("Cargando processor y modelo base")
processor = TrOCRProcessor.from_pretrained(MODELO_BASE)
model = VisionEncoderDecoderModel.from_pretrained(MODELO_BASE)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Penalizar repeticion en generacion: mitiga el colapso visto en el
# modelo anterior (ej. "F7R701" -> "FRRRRRRRRR701")
model.generation_config.no_repeat_ngram_size = 2
model.generation_config.repetition_penalty = 1.3
model.generation_config.max_length = GEN_MAX_LENGTH

train_dataset = PlacasDataset(
    registros_train, processor, MAX_LABEL_LEN, augment=True, n_aug=AUGMENT_N
)
eval_dataset = PlacasDataset(
    registros_val, processor, MAX_LABEL_LEN, augment=False, n_aug=0
)

print(f"  Train final (con augmentacion x{AUGMENT_N + 1}): {len(train_dataset)}")
print(f"  Validation final (sin augmentar): {len(eval_dataset)}")

# =============================================================================
# ENTRENAMIENTO
# =============================================================================

write("Configurando entrenamiento")
args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True,
    generation_max_length=GEN_MAX_LENGTH,
    logging_dir=LOG_DIR,
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
)

ultimo_checkpoint = None
if os.path.isdir(CHECKPOINT_DIR):
    candidatos = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
    candidatos = [d for d in candidatos
                  if os.path.exists(os.path.join(CHECKPOINT_DIR, d, "optimizer.pt"))]
    if candidatos:
        candidatos.sort(key=lambda d: int(d.split("-")[1]))
        ultimo_checkpoint = os.path.join(CHECKPOINT_DIR, candidatos[-1])

write("Iniciando entrenamiento" if not ultimo_checkpoint
      else f"Reanudando entrenamiento desde {ultimo_checkpoint}")
trainer.train(resume_from_checkpoint=ultimo_checkpoint)

write("Guardando modelo final")
os.makedirs(MODEL_OUT, exist_ok=True)
model.save_pretrained(MODEL_OUT)
processor.save_pretrained(MODEL_OUT)

write("Guardando historial de entrenamiento")
df_hist = pd.DataFrame(trainer.state.log_history)
df_hist.to_csv(os.path.join(BASE_DIR, "training_history.csv"), index=False)

write(f"Entrenamiento finalizado. Modelo guardado en: {MODEL_OUT}")
