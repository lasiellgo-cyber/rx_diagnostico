import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
from PIL import Image
import torchxrayvision as xrv

# ── CONFIGURACIÓN CORREGIDA ──
DEVICE = "cpu" 
MODEL_CACHE_PATH = "/tmp/densenet_finetuned.pth"
# ENLACE CORREGIDO PARA DESCARGA DIRECTA:
HF_MODEL_URL = "https://huggingface.co/LASIELL/rx-modelo/resolve/main/densenet_finetuned.pth?download=true"

CATEGORIAS = [
    "Atelectasia", "Cardiomegalia", "Efusión", "Infiltración", "Masa",
    "Nódulo", "Neumonía", "Neumotórax", "Consolidación", "Edema",
    "Enfisema", "Fibrosis", "Engrosamiento Pleural", "Hernia"
]

st.set_page_config(page_title="RX Diagnóstico IA", page_icon="🔬", layout="wide")

@st.cache_resource
def descargar_y_cargar_modelo():
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    modelo.op_threshs = None
    modelo.classifier = nn.Linear(modelo.classifier.in_features, 14)
    
    tipo = "BASE"
    # Forzar descarga si el archivo no existe o es demasiado pequeño (error previo)
    if not os.path.exists(MODEL_CACHE_PATH) or os.path.getsize(MODEL_CACHE_PATH) < 1000000:
        try:
            urllib.request.urlretrieve(HF_MODEL_URL, MODEL_CACHE_PATH)
        except Exception: pass

    if os.path.exists(MODEL_CACHE_PATH):
        try:
            # Cargamos su entrenamiento de 0.11/0.15 loss
            modelo.load_state_dict(torch.load(MODEL_CACHE_PATH, map_location="cpu"), strict=False)
            tipo = "ENTRENADO (RUBÉN)"
        except Exception: pass
            
    return modelo.eval(), tipo

# ... (El resto de su código v5 de visualización sigue igual)