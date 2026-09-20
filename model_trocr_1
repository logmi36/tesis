import os
import torch
from datasets import load_dataset, Dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import sqlite3
import datetime
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from transformers import TrOCRProcessor, VisionEncoderDecoderModel



def show(list):
    dt=datetime.datetime.now()
    dtt=dt.strftime("%H%M%S")
    print(dtt,*list, sep='\t')


def write(cad):
    dt=datetime.datetime.now()
    dtt=dt.strftime("%H%M%S")
    print('\n')
    print('='*80)
    print(dtt,'\t',cad,' ======')
    print('\n')



os.environ["WANDB_DISABLED"]="true"

conn = sqlite3.connect("camera.sqlite")
cursor = conn.cursor()

carpeta_imagenes=r"C:\Users\pc\Documents\camera\plates_mod"


write('init')

sql="select txt2 as archivo, rem as texto from plates where kd='train'"
cursor.execute(sql)

df = pd.read_sql(sql, conn)


write('CONFIGURACIÓN ---')

modelo_preentrenado = "microsoft/trocr-base-printed"

write('--- CARGA DEL DATASET ---')
nombres, textos = [], []

nombres=df["archivo"].to_list()
textos=df["texto"].to_list()

write('Crear dataset HuggingFace')
datos = Dataset.from_dict({
    "image_path": [os.path.join(carpeta_imagenes, n) for n in nombres],
    "text": textos
})

write('Cargar modelo y processor')
#processor = TrOCRProcessor.from_pretrained(modelo_preentrenado)
#model = VisionEncoderDecoderModel.from_pretrained(modelo_preentrenado)


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id


# Asegurarse de que el tokenizer tiene un pad_token
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

model.config.pad_token_id = processor.tokenizer.pad_token_id



write('Preprocesamiento')
def cargar_y_preparar(ejemplo):
    imagen = Image.open(ejemplo["image_path"]).convert("RGB")
    pixel_values = processor(images=imagen, return_tensors="pt").pixel_values[0]
    labels = processor.tokenizer(ejemplo["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=32).input_ids[0]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": labels}

datos = datos.map(cargar_y_preparar)

write('Dividir en entrenamiento y prueba')
datos = datos.train_test_split(test_size=0.2)
train_dataset = datos["train"]
eval_dataset = datos["test"]

#3 eval3
#eval 3 es el que se usara para la tesis!

args = Seq2SeqTrainingArguments(
    output_dir="./trocr_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    num_train_epochs=15,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True,
    generation_max_length=15,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=3,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=False
)


write(' --- FUNCIONES AUXILIARES ---')
def collate_fn(batch):
    pixel_values = [torch.tensor(ej["pixel_values"]) if not isinstance(ej["pixel_values"], torch.Tensor) else ej["pixel_values"] for ej in batch]
    labels = [torch.tensor(ej["labels"]) if not isinstance(ej["labels"], torch.Tensor) else ej["labels"] for ej in batch]

    pixel_values = torch.stack(pixel_values)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

write(' --- ENTRENADOR ---')
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=collate_fn,
)

write(' --- ENTRENAMIENTO ---')
trainer.train()

write(' --- save ---')
model.save_pretrained("trocr_model_final_5")
processor.save_pretrained("trocr_model_final_5")

write('end')
