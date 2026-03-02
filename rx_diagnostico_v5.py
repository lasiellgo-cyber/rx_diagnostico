import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
from PIL import Image
import torchxrayvision as xrv

# ── CONFIGURACIÓN TÉCNICA ──
DEVICE = "cpu" 
MODEL_CACHE_PATH = "/tmp/densenet_finetuned.pth"
# ENLACE CORREGIDO (Descarga directa)
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
    # Forzar descarga limpia
    if not os.path.exists(MODEL_CACHE_PATH) or os.path.getsize(MODEL_CACHE_PATH) < 1000000:
        try:
            urllib.request.urlretrieve(HF_MODEL_URL, MODEL_CACHE_PATH)
        except Exception: pass

    if os.path.exists(MODEL_CACHE_PATH):
        try:
            modelo.load_state_dict(torch.load(MODEL_CACHE_PATH, map_location="cpu"), strict=False)
            tipo = "RUBÉN (ENTRENADO)"
        except Exception: pass
            
    return modelo.eval(), tipo

# ── INTERFAZ ──
st.title("🔬 Sistema de Diagnóstico RX")
modelo, tipo_ia = descargar_y_cargar_modelo()

archivo = st.file_uploader("Cargar imagen RX", type=["jpg","jpeg","png"])
if archivo:
    col1, col2 = st.columns(2)
    img_pil = Image.open(archivo).convert("L")
    with col1:
        st.image(img_pil, use_container_width=True)
        st.caption(f"Modelo activo: {tipo_ia}")
    
    with col2:
        # Procesamiento
        img = np.array(img_pil)
        img = xrv.datasets.normalize(img, 255)
        t = torch.from_numpy(img[None, None, :, :]).float()
        t = torch.nn.functional.interpolate(t, size=(224, 224))
        
        with torch.no_grad():
            preds = torch.sigmoid(modelo(t)).numpy()[0]
        
        st.subheader("Hallazgos:")
        res = sorted(zip(CATEGORIAS, preds), key=lambda x: -x[1])
        for cat, prob in res:
            if prob > 0.25:
                st.write(f"**{cat}**: {prob*100:.1f}%")
                st.progress(float(prob))