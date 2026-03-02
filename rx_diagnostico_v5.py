import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
from PIL import Image
import torchxrayvision as xrv

# ── CONFIGURACIÓN TÉCNICA ──
DEVICE = "cpu" # En la nube usamos CPU por estabilidad
MODEL_CACHE_PATH = "/tmp/densenet_finetuned.pth"
HF_MODEL_URL = "https://huggingface.co/LASIELL/rx-modelo/resolve/main/densenet_finetuned.pth?download=true"

CATEGORIAS = [
    "Atelectasia", "Cardiomegalia", "Efusión", "Infiltración", "Masa",
    "Nódulo", "Neumonía", "Neumotórax", "Consolidación", "Edema",
    "Enfisema", "Fibrosis", "Engrosamiento Pleural", "Hernia"
]

st.set_page_config(page_title="RX Diagnóstico IA", page_icon="🔬", layout="wide")

# Estilo Profesional (Visor Médico)
st.markdown("""
<style>
    .stApp { background-color: #05080a !important; color: #e0e0e0 !important; }
    .rx-header { padding: 20px; background: #0f172a; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #00d4ff; }
    .result-card { background: #0f172a; border: 1px solid #1e293b; padding: 20px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_ia():
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    modelo.op_threshs = None
    num_ftrs = modelo.classifier.in_features
    modelo.classifier = nn.Linear(num_ftrs, 14)
    
    tipo = "BASE"
    if not os.path.exists(MODEL_CACHE_PATH):
        try:
            urllib.request.urlretrieve(HF_MODEL_URL, MODEL_CACHE_PATH)
        except: pass

    if os.path.exists(MODEL_CACHE_PATH):
        try:
            modelo.load_state_dict(torch.load(MODEL_CACHE_PATH, map_location="cpu"))
            tipo = "ENTRENADO (0.15 LOSS)"
        except: pass
            
    return modelo.eval(), tipo

# ── INTERFAZ ──
st.markdown("""
<div class="rx-header">
    <div style="font-size:1.5rem; font-weight:700; color:#00d4ff;">🔬 RX DIAGNÓSTICO IA v5.2</div>
    <div style="font-size:0.8rem; color:#94a3b8;">SISTEMA DE APOYO AL DIAGNÓSTICO · SCS CANARIAS</div>
</div>
""", unsafe_allow_html=True)

modelo, tipo_ia = cargar_ia()

col1, col2 = st.columns([1, 1.2])

with col1:
    archivo = st.file_uploader("Cargar imagen RX", type=["jpg","jpeg","png"])
    if archivo:
        img = Image.open(archivo)
        st.image(img, use_container_width=True)
        st.caption(f"Modelo activo: {tipo_ia}")

with col2:
    if archivo:
        with st.spinner("Procesando..."):
            # Procesamiento de imagen
            input_img = np.array(Image.open(archivo).convert("L"))
            input_img = xrv.datasets.normalize(input_img, 255)
            input_img = torch.from_numpy(input_img[None, None, :, :]).float()
            input_img = torch.nn.functional.interpolate(input_img, size=(224, 224))
            
            with torch.no_grad():
                feats = modelo.features2(input_img)
                preds = torch.sigmoid(modelo.classifier(feats)).numpy()[0]
        
        st.markdown("### Hallazgos detectados")
        res = sorted(zip(CATEGORIAS, preds), key=lambda x: -x[1])
        for cat, prob in res:
            if prob > 0.25:
                color = "#ff4b4b" if prob > 0.45 else "#ffa500"
                st.write(f"**{cat}**: {prob*100:.1f}%")
                st.progress(float(prob))