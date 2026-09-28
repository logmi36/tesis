"""
Script: train_trocr_v3.py
Descripcion: Fine-tuning CONSERVADOR de TrOCR para reconocimiento de placas
             vehiculares peruanas, partiendo de microsoft/trocr-base-printed.

Motivo (v3 vs v2)
------------------
Los 3 intentos anteriores (den32.py original, v2 desde stage1, v2 desde
trocr-base-printed) EMPEORARON el desempeno del modelo respecto a usarlo
zero-shot (61.56% Exact Match sobre 333 placas de test, sin ningun
entrenamiento). En los 3 casos el eval_loss (teacher-forcing) bajaba de
forma saludable mientras el Exact Match real (generacion libre) caia a
~0%: un caso claro de sesgo de exposicion / sobreajuste, agravado por
ajustar los ~334M parametros del modelo completo con solo 2500 imagenes.

Ajustes de v3
-------------
1. CONGELAMIENTO PARCIAL: se congela todo el encoder de vision (26% de
   los parametros) y las 10 primeras capas del decoder (de 12), dejando
   entrenables solo las ultimas 2 capas del decoder (~10% del modelo).
   Esto preserva casi toda la competencia de OCR ya aprendida, permitiendo
   solo una adaptacion superficial a los caracteres/fuente de las placas.
2. LEARNING RATE MUCHO MAS BAJO (2e-6 vs 3e-5 anterior, ~15x menor), para
   evitar que incluso las capas entrenables se desvien demasiado rapido.
3. SELECCION DE MEJOR CHECKPOINT POR CER REAL (generacion autoregresiva
   via model.generate(), no por eval_loss de teacher-forcing). Esto es
   la causa raiz del problema anterior: eval_loss no se correlacionaba
   con la calidad real de generacion. Ahora se calcula CER y Exact Match
   genuinos en cada epoca (compute_metrics) y esos son el criterio de
   parada temprana / mejor modelo.
4. Pocas epocas (10) con early stopping agresivo (patience=2) sobre CER
   real, dado que un modelo ya competente deberia converger rapido o no
   converger en absoluto (en cuyo caso se detiene pronto sin danar mas).

Autor: Irwin Ivan Mendoza Gonzalez
Tesis: UNI 2025
"""

import os

os.environ["USE_TF"] = "0"

import random
import sqlite3
import datetime
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

# =============================================================================
# CONFIGURACION
# =============================================================================

DB_PATH   = r"C:\Tesis2\camera.sqlite"
IMG_DIRS  = [r"C:\data\media\plates_mod"]

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT      = os.path.join(BASE_DIR, "trocr_model_final_v3")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR        = os.path.join(BASE_DIR, "logs")

MODELO_BASE    = "microsoft/trocr-base-printed"

CAPAS_DECODER_ENTRENABLES = 2   # ultimas N capas del decoder (de 12) que SI se entrenan
EPOCHS         = 10
LEARNING_RATE  = 2e-6
BATCH_SIZE     = 8
MAX_LABEL_LEN  = 16
GEN_MAX_LENGTH = 16
EARLY_STOP_PATIENCE = 2
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


def calcular_cer(referencia, hipotesis):
    m, n = len(referencia), len(hipotesis)
    if m == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if referencia[i - 1] == hipotesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / m


# =============================================================================
# DATASET (carga perezosa, sin augmentacion)
# =============================================================================

class PlacasDataset(torch.utils.data.Dataset):
    def __init__(self, registros, processor, max_label_len):
        self.registros = registros
        self.processor = processor
        self.max_label_len = max_label_len

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, idx):
        ruta, texto = self.registros[idx]
        imagen = Image.open(ruta).convert("RGB")
        pixel_values = self.processor(images=imagen, return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(
            texto, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_label_len,
        ).input_ids[0]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
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
model.generation_config.max_length = GEN_MAX_LENGTH

# =============================================================================
# CONGELAMIENTO PARCIAL
# =============================================================================

write(f"Congelando encoder completo + primeras capas del decoder "
      f"(entrenables solo las ultimas {CAPAS_DECODER_ENTRENABLES} capas del decoder)")

num_capas_decoder = model.config.decoder.decoder_layers  # 12 para trocr-base
capas_congeladas = set(range(0, num_capas_decoder - CAPAS_DECODER_ENTRENABLES))

for name, param in model.named_parameters():
    entrenable = False
    if name.startswith("decoder."):
        m = re.search(r'decoder\.model\.decoder\.layers\.(\d+)\.', name)
        if m and int(m.group(1)) not in capas_congeladas:
            entrenable = True
    param.requires_grad = entrenable

n_total = sum(p.numel() for p in model.parameters())
n_entrenable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parametros totales: {n_total:,}")
print(f"  Parametros entrenables: {n_entrenable:,} ({n_entrenable/n_total*100:.1f}%)")

train_dataset = PlacasDataset(registros_train, processor, MAX_LABEL_LEN)
eval_dataset = PlacasDataset(registros_val, processor, MAX_LABEL_LEN)

# =============================================================================
# METRICA REAL (CER / Exact Match sobre generacion autoregresiva)
# =============================================================================

def compute_metrics(eval_preds):
    pred_ids = eval_preds.predictions
    label_ids = eval_preds.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cers, matches = [], []
    for pred, real in zip(pred_str, label_str):
        pred_n = pred.strip().upper().replace("-", "").replace(" ", "")
        real_n = real.strip().upper().replace("-", "").replace(" ", "")
        cers.append(calcular_cer(real_n, pred_n))
        matches.append(1.0 if pred_n == real_n else 0.0)

    return {
        "cer": float(np.mean(cers)),
        "exact_match": float(np.mean(matches)) * 100,
    }


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
    metric_for_best_model="cer",
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
    compute_metrics=compute_metrics,
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

write("Guardando modelo final (mejor checkpoint segun CER real)")
os.makedirs(MODEL_OUT, exist_ok=True)
model.save_pretrained(MODEL_OUT)
processor.save_pretrained(MODEL_OUT)

write("Guardando historial de entrenamiento")
df_hist = pd.DataFrame(trainer.state.log_history)
df_hist.to_csv(os.path.join(BASE_DIR, "training_history.csv"), index=False)

write(f"Entrenamiento finalizado. Modelo guardado en: {MODEL_OUT}")
