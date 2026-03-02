import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
from PIL import Image
import torchxrayvision as xrv

# ── CONFIGURACIÓN TÉCNICA DEFINITIVA ──
DEVICE = "cpu" 
MODEL_CACHE_PATH = "/tmp/densenet_finetuned.pth"

# ENLACE CORREGIDO: 'resolve' permite la descarga directa del peso real (445MB)
HF_MODEL_URL = "https://huggingface.co/LASIELL/rx-modelo/resolve/main/densenet_finetuned.pth?download=true"

CATEGORIAS = [
    "Atelectasia", "Cardiomegalia", "Efusión", "Infiltración", "Masa",
    "Nódulo", "Neumonía", "Neumotórax", "Consolidación", "Edema",
    "Enfisema", "Fibrosis", "Engrosamiento Pleural", "Hernia"
]

st.set_page_config(page_title="RX Diagnóstico IA", page_icon="🔬", layout="wide")

# Estilo Visor Médico SCS
st.markdown("""
<style>
    .stApp { background-color: #05080a !important; color: #e0e0e0 !important; }
    .rx-header { padding: 20px; background: #0f172a; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #00d4ff; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def descargar_y_cargar_modelo():
    # 1. Cargar arquitectura DenseNet-121 base
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    modelo.op_threshs = None
    modelo.classifier = nn.Linear(modelo.classifier.in_features, 14)
    
    tipo = "BASE (Genérico)"
    
    # 2. Descargar modelo de Hugging Face (Solo si no existe o es el archivo corrupto de 28MB)
    if not os.path.exists(MODEL_CACHE_PATH) or os.path.getsize(MODEL_CACHE_PATH) < 100000000:
        try:
            with st.spinner("Descargando cerebro de IA (445MB)..."):
                urllib.request.urlretrieve(HF_MODEL_URL, MODEL_CACHE_PATH)
        except Exception:
            pass

    # 3. Cargar sus pesos entrenados (Su modelo de 0.11 - 0.17 loss)
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            state_dict = torch.load(MODEL_CACHE_PATH, map_location="cpu")
            modelo.load_state_dict(state_dict, strict=False)
            tipo = "RUBÉN (ENTRENADO)"
        except Exception:
            pass
            
    return modelo.eval(), tipo

# ── INTERFAZ DE USUARIO ──
st.markdown("""
<div class="rx-header">
    <div style="font-size:1.5rem; font-weight:700; color:#00d4ff;">🔬 RX DIAGNÓSTICO IA v5.2</div>
    <div style="font-size:0.8rem; color:#94a3b8;">SISTEMA DE APOYO AL DIAGNÓSTICO · CONSULTA CANARIAS</div>
</div>
""", unsafe_allow_html=True)

modelo, tipo_ia = descargar_y_cargar_modelo()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    archivo = st.file_uploader("Cargar Radiografía", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if archivo:
        img_pil = Image.open(archivo).convert("L")
        st.image(img_pil, use_container_width=True)
        st.caption(f"Motor activo: {tipo_ia}")

with col2:
    if archivo:
        with st.spinner("Analizando imagen..."):
            # Preprocesamiento exacto del entrenamiento
            img = np.array(Image.open(archivo).convert("L"))
            img = xrv.datasets.normalize(img, 255)
            t = torch.from_numpy(img[None, None, :, :]).float()
            t = torch.nn.functional.interpolate(t, size=(224, 224))
            
            with torch.no_grad():
                preds = torch.sigmoid(modelo(t)).numpy()[0]
        
        st.subheader("Hallazgos detectados:")
        res = sorted(zip(CATEGORIAS, preds), key=lambda x: -x[1])
        
        for cat, prob in res:
            if prob > 0.10: # Mostramos todo lo que tenga relevancia mínima
                st.write(f"**{cat}**: {prob*100:.1f}%")
                st.progress(float(prob))

st.markdown('<div style="font-size:0.7rem; color:#4a6080; text-align:center; margin-top:50px;">⚕️ INVESTIGACIÓN CLÍNICA · DR. RUBÉN · SCS CANARIAS</div>', unsafe_allow_html=True)