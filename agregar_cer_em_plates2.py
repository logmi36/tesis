"""
Script: agregar_cer_em_plates2.py
Descripcion: Agrega a la tabla `plates2` de camera2.sqlite columnas con el
             CER y el Exact Match (EM) individual de cada una de las 7
             configuraciones evaluadas del modelo TrOCR, usando como llave
             de union el nombre de archivo de la placa recortada (name3).

Columnas agregadas (14 en total, REAL para cer_*, INTEGER 0/1 para em_*):
    cer_e1, em_e1           Estrategia 1 (fine-tuning completo, stage1)
    cer_e1aug, em_e1aug     Estrategia 1-aumentada
    cer_e2, em_e2           Estrategia 2 (fine-tuning completo, printed)
    cer_e2aug, em_e2aug     Estrategia 2-aumentada
    cer_e3, em_e3           Estrategia 3 (fine-tuning conservador)
    cer_e3aug, em_e3aug     Estrategia 3-aumentada
    cer_zeroshot, em_zeroshot   Estrategia 4 (zero-shot, adoptada)

Solo las 333 placas del split kd='test' reciben valores; el resto queda NULL
(no fueron evaluadas con estos modelos).

Autor: Irwin Ivan Mendoza Gonzalez
Tesis: UNI 2025
"""

import sqlite3
import csv

DB_PATH = r"C:\data\media\camera2.sqlite"

ESTRATEGIAS = [
    ("e1",        r"C:\data\media\trocr_v2\results_trocr_v1_stage1.csv"),
    ("e1aug",     r"C:\data\media\trocr_v1_aug\results_trocr_v1_aug.csv"),
    ("e2",        r"C:\data\media\trocr_v2\results_trocr_v2.csv"),
    ("e2aug",     r"C:\data\media\trocr_v2_aug\results_trocr_v2_aug.csv"),
    ("e3",        r"C:\data\media\trocr_v3\results_trocr_v3.csv"),
    ("e3aug",     r"C:\data\media\trocr_v3_aug\results_trocr_v3_aug.csv"),
    ("zeroshot",  r"C:\data\media\trocr_v2\results_trocr_zeroshot.csv"),
]


def leer_resultados(ruta):
    """Devuelve dict: archivo -> (cer, em)"""
    out = {}
    with open(ruta, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            archivo = row["archivo"].strip()
            cer = float(row["cer"])
            match_raw = row["match"].strip().lower()
            em = 1 if match_raw == "true" else 0
            out[archivo] = (cer, em)
    return out


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Agregar columnas (si no existen ya)
    cur.execute("PRAGMA table_info(plates2)")
    columnas_existentes = {row[1] for row in cur.fetchall()}

    for sufijo, _ in ESTRATEGIAS:
        for prefijo, tipo in [("cer", "REAL"), ("em", "INTEGER")]:
            col = f"{prefijo}_{sufijo}"
            if col not in columnas_existentes:
                cur.execute(f'ALTER TABLE plates2 ADD COLUMN "{col}" {tipo}')
                print(f"  Columna agregada: {col} ({tipo})")
            else:
                print(f"  Columna ya existia: {col}")
    conn.commit()

    # 2. Poblar valores
    for sufijo, ruta_csv in ESTRATEGIAS:
        resultados = leer_resultados(ruta_csv)
        col_cer = f"cer_{sufijo}"
        col_em = f"em_{sufijo}"

        actualizados = 0
        no_encontrados = []
        for archivo, (cer, em) in resultados.items():
            cur.execute(
                f'UPDATE plates2 SET "{col_cer}" = ?, "{col_em}" = ? WHERE name3 = ?',
                (cer, em, archivo),
            )
            if cur.rowcount == 0:
                no_encontrados.append(archivo)
            else:
                actualizados += cur.rowcount

        print(f"[{sufijo}] filas actualizadas: {actualizados} / {len(resultados)}"
              + (f"  (sin match: {len(no_encontrados)})" if no_encontrados else ""))
        if no_encontrados:
            print("    ejemplos sin match:", no_encontrados[:5])

    conn.commit()

    # 3. Verificacion final
    cur.execute("SELECT COUNT(*) FROM plates2 WHERE cer_zeroshot IS NOT NULL")
    print("\nTotal de filas con valores de zero-shot pobladas:", cur.fetchone()[0])

    conn.close()
    print("\nListo.")


if __name__ == "__main__":
    main()
