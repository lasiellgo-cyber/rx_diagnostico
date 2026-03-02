"""
SISTEMA DE DIAGNÓSTICO IA - DR. RUBÉN
"""
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
from PIL import Image
import torchxrayvision as xrv

# ── CONFIGURACIÓN DE RUTAS Y ENLACES ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE_PATH = "/tmp/densenet_finetuned.pth"
HF_MODEL_URL = "https://huggingface.co/LASIELL/rx-modelo/resolve/main/densenet_finetuned.pth?download=true"

LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "modelos_entrenados", "densenet_finetuned.pth")

CATEGORIAS = [
    "Atelectasia", "Cardiomegalia", "Efusión", "Infiltración", "Masa",
    "Nódulo", "Neumonía", "Neumotórax", "Consolidación", "Edema",
    "Enfisema", "Fibrosis", "Engrosamiento Pleural", "Hernia"
]

ZONA_ANATOMICA = {
    "Atelectasia": "Pulmón (lóbulo inferior)", "Cardiomegalia": "Corazón / Mediastino",
    "Efusión": "Espacio pleural", "Infiltración": "Parénquima pulmonar",
    "Masa": "Pulmón / Mediastino", "Nódulo": "Parénquima pulmonar",
    "Neumonía": "Pulmón (consolidación)", "Neumotórax": "Espacio pleural / Apex",
    "Consolidación": "Parénquima pulmonar", "Edema": "Pulmón bilateral / Hilio",
    "Enfisema": "Pulmón (hiperinsuflación)", "Fibrosis": "Intersticio pulmonar",
    "Engrosamiento Pleural": "Pleura", "Hernia": "Diafragma / Mediastino inf."
}

HALLAZGO_VISUAL = {
    "Atelectasia": "Opacidad laminar con pérdida de volumen",
    "Cardiomegalia": "Índice cardiotorácico > 0.5, silueta cardíaca aumentada",
    "Efusión": "Opacificación del seno costofrénico, menisco pleural",
    "Infiltración": "Opacidades heterogéneas de predominio peribronquial",
    "Masa": "Opacidad redondeada > 3 cm con bordes bien definidos",
    "Nódulo": "Opacidad redondeada < 3 cm",
    "Neumonía": "Consolidación alveolar con broncograma aéreo",
    "Neumotórax": "Línea pleural visible, ausencia de trama vascular",
    "Consolidación": "Opacidad homogénea con broncograma aéreo",
    "Edema": "Opacidades bilaterales perihiliares, líneas B de Kerley",
    "Enfisema": "Hiperinsuflación y aplanamiento diafragmático",
    "Fibrosis": "Opacidades reticulares basales, patrón en panal",
    "Engrosamiento Pleural": "Opacidad pleural periférica irregular",
    "Hernia": "Estructura abdominal por encima del diafragma",
}

UMBRAL_POSITIVO = 0.45
UMBRAL_SUGESTIVO = 0.25

st.set_page_config(page_title="IA Diagnóstica - Dr. Rubén", page_icon="🔬", layout="wide")

# Estilo CSS Profesional (Visor de Grado Médico)
st.markdown("""
<style>
    :root { --bg:#020508; --card:#0a0e14; --accent:#00d4ff; --green:#00c853; --red:#d50000; --text:#e0e0e0; }
    .stApp { background: var(--bg) !important; color: var(--text) !important; }
    .rx-header { display:flex; align-items:center; gap:20px; padding:25px; background:var(--card); border-radius:4px; margin-bottom:30px; border-bottom:2px solid #1a202c; }
    .result-anormal { background:rgba(213,0,0,0.05); border:1px solid var(--red); border-radius:4px; padding:25px; }
    .result-normal { background:rgba(0,200,83,0.05); border:1px solid var(--green); border-radius:4px; padding:25px; }
    .zona-tag { display:inline-block; border:1px solid #1e293b; color:#94a3b8; padding:3px 8px; border-radius:2px; margin-right:5px; font-family:monospace; font-size:0.75rem; }
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; letter-spacing: -0.5px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def descargar_y_cargar_modelo():
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    modelo.op_threshs = None
    num_ftrs = modelo.classifier.in_features
    modelo.classifier = nn.Linear(num_ftrs, 14)
    
    tipo = "BASE"
    path_final = None

    if os.path.exists(LOCAL_MODEL_PATH):
        path_final = LOCAL_MODEL_PATH
    else:
        if not os.path.exists(MODEL_CACHE_PATH) or os.path.getsize(MODEL_CACHE_PATH) < 10000:
            try:
                urllib.request.urlretrieve(HF_MODEL_URL, MODEL_CACHE_PATH)
                st.toast("Cerebro de IA Rubén cargado", icon="🔬")
            except Exception:
                pass
        if os.path.exists(MODEL_CACHE_PATH):
            path_final = MODEL_CACHE_PATH

    if path_final:
        try:
            state = torch.load(path_final, map_location="cpu")
            modelo.load_state_dict(state, strict=False)
            tipo = "RUBÉN (0.15 LOSS)"
        except Exception:
            pass
            
    modelo.to(DEVICE).eval()
    return modelo, tipo

def analizar(imagen_pil, modelo):
    img = np.array(imagen_pil.convert("L"))
    img = xrv.datasets.normalize(img, 255)
    t = torch.from_numpy(img[None, None, :, :]).float().to(DEVICE)
    t = torch.nn.functional.interpolate(t, size=(224, 224))
    with torch.no_grad():
        feats = modelo.features2(t)
        preds = torch.sigmoid(modelo.classifier(feats)).cpu().numpy()[0]
    return preds

# ── INTERFAZ ──
st.markdown(f"""
<div class="rx-header">
    <div style="font-size:2.2rem;">🔬</div>
    <div>
        <div style="font-family:monospace; font-size:0.7rem; color:var(--accent); letter-spacing:4px; margin-bottom:4px;">PROYECTO MÉDICO</div>
        <div style="font-size:1.6rem; color:white; font-weight:700;">RUBÉN · CONSULTA PRIVADA</div>
    </div>
</div>
""", unsafe_allow_html=True)

modelo,