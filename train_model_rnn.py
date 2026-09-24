"""
Script: train_model_rnn.py
Descripción: Entrena y guarda el modelo RNN (LSTM) para reconocimiento
             de caracteres de placas vehiculares peruanas.

Autor: Irwin Ivan Mendoza Gonzalez
Tesis: UNI 2025
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CHARS_DIR  = r"C:\data\media\chars"
MODELS_DIR = r"C:\data\media\models"

CHAR_SIZE      = (28, 28)
RNN_NEURONAS   = 128
RNN_EPOCAS     = 40
BATCH_SIZE     = 32
VAL_SPLIT      = 0.15
CHARS_VALIDOS  = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
AUGMENT_N      = 4  # variaciones sintéticas por imagen (0 = sin augmentación)

np.random.seed(42)

# =============================================================================
# AUMENTO DE DATOS
# =============================================================================

def augmentar_caracter(img, n_aug=AUGMENT_N):
    """
    Genera variaciones sintéticas de un carácter (rotación, traslación,
    grosor de trazo y ruido) para reducir la brecha entre las imágenes
    limpias de entrenamiento y los recortes reales de campo.
    """
    h, w = img.shape
    variantes = [img]
    for _ in range(n_aug):
        aug = img.copy()

        angulo = np.random.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderValue=0)

        tx, ty = np.random.randint(-2, 3, size=2)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        aug = cv2.warpAffine(aug, M, (w, h), borderValue=0)

        k = np.random.choice([0, 1, 1, 2])
        if k > 0:
            kernel = np.ones((k, k), np.uint8)
            aug = cv2.dilate(aug, kernel, iterations=1) \
                if np.random.rand() < 0.5 else cv2.erode(aug, kernel, iterations=1)

        if np.random.rand() < 0.3:
            ruido = np.random.normal(0, 10, aug.shape).astype(np.float32)
            aug = np.clip(aug.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

        variantes.append(aug)
    return variantes


def augmentar_dataset(X, y):
    """Aplica augmentar_caracter() a cada imagen y expande X, y."""
    X_aug, y_aug = [], []
    for img, label in zip(X, y):
        for variante in augmentar_caracter(img):
            X_aug.append(variante)
            y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_dataset():
    """Carga las imágenes originales (sin augmentar) en uint8."""
    print("\n📂 Cargando dataset de caracteres...")
    X, y = [], []

    for char in CHARS_VALIDOS:
        carpeta = os.path.join(CHARS_DIR, char)
        if not os.path.exists(carpeta):
            continue
        for img_file in os.listdir(carpeta):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(os.path.join(carpeta, img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, CHAR_SIZE)
            X.append(img)
            y.append(char)

    X = np.array(X, dtype='uint8')
    y = np.array(y)
    print(f"  ✅ Total imágenes originales: {len(X)}")
    return X, y

# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def main():
    print("=" * 50)
    print("ENTRENAMIENTO — MODELO RNN (LSTM)")
    print("Tesis UNI 2025")
    print("=" * 50)

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    # Cargar datos originales
    X, y = cargar_dataset()

    # Codificar etiquetas
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_clases = len(le.classes_)

    # Dividir train/test ANTES de aumentar (evita fuga de variantes
    # sintéticas del mismo carácter entre train y test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=VAL_SPLIT, random_state=42, stratify=y_enc
    )

    # Aumentar solo el conjunto de entrenamiento
    print(f"\n🔀 Aumentando datos de entrenamiento (x{AUGMENT_N + 1})...")
    X_train, y_train = augmentar_dataset(X_train, y_train)

    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32') / 255.0

    print(f"  ✅ Train: {len(X_train)} | Test: {len(X_test)}")

    # Reshape para RNN (samples, timesteps, features)
    X_train_rnn = X_train.reshape(len(X_train), CHAR_SIZE[0], CHAR_SIZE[1])
    X_test_rnn  = X_test.reshape(len(X_test), CHAR_SIZE[0], CHAR_SIZE[1])

    # One-hot encoding
    y_train_oh = tf.keras.utils.to_categorical(y_train, n_clases)
    y_test_oh  = tf.keras.utils.to_categorical(y_test, n_clases)

    # Arquitectura RNN LSTM
    print(f"\n🔧 Entrenando RNN LSTM ({RNN_NEURONAS} neuronas, {RNN_EPOCAS} épocas)...")
    modelo = models.Sequential([
        layers.LSTM(RNN_NEURONAS,
                    input_shape=(CHAR_SIZE[0], CHAR_SIZE[1]),
                    return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_clases, activation='softmax')
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    modelo.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Entrenar
    history = modelo.fit(
        X_train_rnn, y_train_oh,
        epochs=RNN_EPOCAS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluar
    _, acc = modelo.evaluate(X_test_rnn, y_test_oh, verbose=0)
    print(f"\n✅ Accuracy RNN: {acc*100:.2f}%")

    # Guardar modelo
    ruta_modelo = os.path.join(MODELS_DIR, 'rnn_model.h5')
    modelo.save(ruta_modelo)
    print(f"\n📁 Modelo guardado en: {ruta_modelo}")

    # Guardar historial
    import pandas as pd
    df_hist = pd.DataFrame(history.history)
    df_hist.to_csv(os.path.join(MODELS_DIR, 'rnn_history.csv'), index=False)
    print(f"📁 Historial guardado en: {MODELS_DIR}/rnn_history.csv")


if __name__ == "__main__":
    main()
