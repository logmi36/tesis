import pandas as pd
import shutil
import os
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN — Modifica estas rutas según tu sistema
# =============================================================================

# Ruta del archivo plates.csv
CSV_PATH = r"C:\data\media\plates.csv"

# Carpeta con imágenes originales (VI2 = -1)
PLATES_DIR = r"C:\data\media\groups"

# Carpeta con imágenes preprocesadas (VI2 = +1)
PLATES_MOD_DIR = r"C:\data\media\plates_mod"

# Carpeta raíz donde se generarán los 4 conjuntos
OUTPUT_DIR = r"C:\data\media\conjuntos_factorial"

# Separador del CSV (verificar si es ; o ,)
CSV_SEPARATOR = ';'

# Nombre de columnas en el CSV
COL_ARCHIVO = 'archivo'
COL_TIPO = 'tipo'
COL_TEXTO = 'texto placa'
COL_ESTADO = 'Estado Fisico Placa'
COL_ANGULO = 'Angulo de captura'
COL_LUZ = 'condiciones de Luz'

# =============================================================================
# REGLA DE CLASIFICACIÓN VI1
# Sección 6.6.3 de la tesis
# =============================================================================

# Valores favorables por dimensión
FAVORABLES_LUZ = [
    'Día con luz solar directa',
    'Sombra parcial',
    'Día nublado / luz difusa'
]

FAVORABLES_ANGULO = [
    'Frontal recto (perpendicular)',
    'Trasera vertical (cámara alta mirando abajo)',
    'Trasera recta'
]

FAVORABLES_ESTADO = [
    'Limpia'
]


def clasificar_vi1(row):
    """
    Regla combinada OR para VI1:
    VI1 = +1 si las 3 dimensiones son favorables simultáneamente
    VI1 = -1 si al menos una dimensión es desfavorable
    """
    luz_favorable = str(row[COL_LUZ]).strip() in FAVORABLES_LUZ
    angulo_favorable = str(row[COL_ANGULO]).strip() in FAVORABLES_ANGULO
    estado_favorable = str(row[COL_ESTADO]).strip() in FAVORABLES_ESTADO

    if luz_favorable and angulo_favorable and estado_favorable:
        return 1   # Favorable
    else:
        return -1  # Desfavorable


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def crear_conjunto(nombre, imagenes, texto_placa, carpeta_origen, carpeta_destino):
    """
    Copia las imágenes del conjunto a su carpeta destino y
    genera un CSV con el ground truth (texto real de la placa).
    """
    # Crear carpeta destino
    Path(carpeta_destino).mkdir(parents=True, exist_ok=True)

    copiadas = 0
    no_encontradas = 0

    for archivo, texto in zip(imagenes, texto_placa):
        origen = os.path.join(carpeta_origen, archivo)
        destino = os.path.join(carpeta_destino, archivo)

        if os.path.exists(origen):
            shutil.copy2(origen, destino)
            copiadas += 1
        else:
            print(f"  ⚠️  No encontrada: {archivo}")
            no_encontradas += 1

    # Guardar CSV con ground truth del conjunto
    df_conjunto = pd.DataFrame({
        'archivo': imagenes,
        'texto_placa': texto_placa
    })
    csv_destino = os.path.join(carpeta_destino, f'ground_truth_{nombre}.csv')
    df_conjunto.to_csv(csv_destino, index=False, sep=';')

    return copiadas, no_encontradas


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

def main():
    print("=" * 65)
    print("GENERACIÓN DE CONJUNTOS DE EVALUACIÓN — DISEÑO FACTORIAL")
    print("VI1 × VI2 → 4 conjuntos de evaluación")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # PASO 1: Leer el CSV
    # -------------------------------------------------------------------------
    print("\n📂 Paso 1: Leyendo plates.csv...")

    try:
        df = pd.read_csv(CSV_PATH, sep=CSV_SEPARATOR, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, sep=CSV_SEPARATOR, encoding='latin-1')

    print(f"  ✅ Total de registros: {len(df)}")
    print(f"  ✅ Columnas: {list(df.columns)}")

    # -------------------------------------------------------------------------
    # PASO 2: Filtrar conjunto de prueba
    # -------------------------------------------------------------------------
    print("\n📂 Paso 2: Filtrando conjunto de prueba (tipo=test)...")

    df_test = df[df[COL_TIPO] == 'test'].copy()
    print(f"  ✅ Imágenes de prueba: {len(df_test)}")

    # -------------------------------------------------------------------------
    # PASO 3: Clasificar VI1
    # -------------------------------------------------------------------------
    print("\n📂 Paso 3: Clasificando imágenes según VI1...")

    df_test['vi1'] = df_test.apply(clasificar_vi1, axis=1)

    n_favorable = len(df_test[df_test['vi1'] == 1])
    n_desfavorable = len(df_test[df_test['vi1'] == -1])

    print(f"  ✅ VI1 = +1 (favorable):    {n_favorable} imágenes ({n_favorable/len(df_test)*100:.1f}%)")
    print(f"  ✅ VI1 = -1 (desfavorable): {n_desfavorable} imágenes ({n_desfavorable/len(df_test)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # PASO 4: Separar en subconjuntos según VI1
    # -------------------------------------------------------------------------
    df_favorable = df_test[df_test['vi1'] == 1]
    df_desfavorable = df_test[df_test['vi1'] == -1]

    archivos_fav = df_favorable[COL_ARCHIVO].tolist()
    textos_fav = df_favorable[COL_TEXTO].tolist()

    archivos_desfav = df_desfavorable[COL_ARCHIVO].tolist()
    textos_desfav = df_desfavorable[COL_TEXTO].tolist()

    # -------------------------------------------------------------------------
    # PASO 5: Generar los 4 conjuntos
    # -------------------------------------------------------------------------
    print("\n📂 Paso 4: Generando los 4 conjuntos de evaluación...")

    conjuntos = [
        {
            'nombre': 'A',
            'vi1': '+1 (favorable)',
            'vi2': '+1 (con preprocesamiento)',
            'archivos': archivos_fav,
            'textos': textos_fav,
            'carpeta_origen': PLATES_MOD_DIR,
            'carpeta_destino': os.path.join(OUTPUT_DIR, 'conjunto_A_fav_preprocesado')
        },
        {
            'nombre': 'B',
            'vi1': '+1 (favorable)',
            'vi2': '-1 (sin preprocesamiento)',
            'archivos': archivos_fav,
            'textos': textos_fav,
            'carpeta_origen': PLATES_DIR,
            'carpeta_destino': os.path.join(OUTPUT_DIR, 'conjunto_B_fav_original')
        },
        {
            'nombre': 'C',
            'vi1': '-1 (desfavorable)',
            'vi2': '+1 (con preprocesamiento)',
            'archivos': archivos_desfav,
            'textos': textos_desfav,
            'carpeta_origen': PLATES_MOD_DIR,
            'carpeta_destino': os.path.join(OUTPUT_DIR, 'conjunto_C_desfav_preprocesado')
        },
        {
            'nombre': 'D',
            'vi1': '-1 (desfavorable)',
            'vi2': '-1 (sin preprocesamiento)',
            'archivos': archivos_desfav,
            'textos': textos_desfav,
            'carpeta_origen': PLATES_DIR,
            'carpeta_destino': os.path.join(OUTPUT_DIR, 'conjunto_D_desfav_original')
        }
    ]

    print()
    for c in conjuntos:
        print(f"  Generando Conjunto {c['nombre']} "
              f"(VI1={c['vi1']}, VI2={c['vi2']})...")

        copiadas, errores = crear_conjunto(
            c['nombre'],
            c['archivos'],
            c['textos'],
            c['carpeta_origen'],
            c['carpeta_destino']
        )

        print(f"  ✅ {copiadas} imágenes copiadas | "
              f"❌ {errores} no encontradas")
        print(f"  📁 → {c['carpeta_destino']}\n")

    # -------------------------------------------------------------------------
    # RESUMEN FINAL
    # -------------------------------------------------------------------------
    print("=" * 65)
    print("RESUMEN DE CONJUNTOS GENERADOS")
    print("=" * 65)
    print(f"\n{'Conjunto':<12} {'VI1':<20} {'VI2':<25} {'N° imágenes'}")
    print("-" * 65)
    print(f"{'A':<12} {'+1 favorable':<20} {'+1 con preprocesamiento':<25} {n_favorable}")
    print(f"{'B':<12} {'+1 favorable':<20} {'-1 sin preprocesamiento':<25} {n_favorable}")
    print(f"{'C':<12} {'-1 desfavorable':<20} {'+1 con preprocesamiento':<25} {n_desfavorable}")
    print(f"{'D':<12} {'-1 desfavorable':<20} {'-1 sin preprocesamiento':<25} {n_desfavorable}")
    print("-" * 65)
    print(f"{'Total':<12} {'':<20} {'':<25} {(n_favorable*2 + n_desfavorable*2)}")

    print(f"\n📁 Conjuntos guardados en:")
    print(f"   {OUTPUT_DIR}")
    print(f"\n✅ Cada conjunto incluye un archivo ground_truth_X.csv")
    print(f"   con los textos reales de las placas para evaluación.")
    print("\n🎯 Siguiente paso: correr los scripts de evaluación de")
    print("   cada modelo sobre los 4 conjuntos generados.")


if __name__ == "__main__":
    main()
