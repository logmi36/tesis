"""
Script: eval_trocr_zeroshot.py
Descripcion: Evalua microsoft/trocr-base-printed SIN fine-tuning (zero-shot)
             sobre el split de TEST de la tabla `plates` (333 imagenes).

Resultado de esta evaluacion: el modelo recomendado final para la tesis,
tras confirmar que 3 intentos de fine-tuning (den32.py original, v2 desde
stage1, v2 desde printed) empeoraron el desempeno zero-shot del modelo
base en vez de mejorarlo.

Autor: Irwin Ivan Mendoza Gonzalez
Tesis: UNI 2025
"""

import os

os.environ["USE_TF"] = "0"

import sqlite3
import datetime

import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DB_PATH   = r"C:\Tesis2\camera.sqlite"
IMG_DIRS  = [r"C:\data\media\plates_mod"]

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_ID   = "microsoft/trocr-base-printed"
OUTPUT_CSV = os.path.join(BASE_DIR, "results_trocr_zeroshot.csv")


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


def main():
    write(f"Cargando {MODEL_ID} (zero-shot, sin fine-tuning)")
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.eval()

    conn = sqlite3.connect(DB_PATH)
    df_test = pd.read_sql("select txt2 as archivo, rem as texto from plates where kd='test'", conn)
    conn.close()

    write(f"Evaluando sobre {len(df_test)} placas de test")
    resultados = []
    for _, row in df_test.iterrows():
        archivo = row["archivo"]
        real = str(row["texto"]).strip().upper().replace("-", "")
        ruta = buscar_imagen(archivo)
        if ruta is None:
            resultados.append({"archivo": archivo, "real": real, "pred": "??????", "cer": 1.0, "match": False})
            continue

        img = Image.open(ruta).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=16)
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred = pred.strip().upper().replace("-", "").replace(" ", "")

        cer = calcular_cer(real, pred)
        match = real == pred
        resultados.append({"archivo": archivo, "real": real, "pred": pred, "cer": cer, "match": match})
        print(f"{archivo}\treal={real}\tpred={pred}\tmatch={match}\tcer={cer:.3f}")

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(OUTPUT_CSV, index=False, sep=";")

    cer_prom = df_res["cer"].mean()
    em = df_res["match"].mean() * 100
    write("RESUMEN FINAL - trocr-base-printed ZERO-SHOT sobre TEST (n=%d)" % len(df_res))
    print(f"CER promedio: {cer_prom:.4f}")
    print(f"Exact Match: {em:.2f}%")
    print(f"\nResultados guardados en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
