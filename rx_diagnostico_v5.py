"""
RX DIAGNÓSTICO IA v5.0
======================
- Interfaz médica profesional
- Análisis automático al subir imagen
- 14 patologías NIH con barras de probabilidad
- Modelo TorchXRayVision DenseNet-121
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchxrayvision as xrv

DEVICE = "cpu"

CATEGORIAS = [
    "Atelectasia", "Cardiomegalia", "Efusión pleural", "Infiltración",
    "Masa torácica", "Nódulo pulmonar", "Neumonía", "Neumotórax",
    "Consolidación", "Edema pulmonar", "Enfisema", "Fibrosis",
    "Engrosamiento pleural", "Hernia"
]

st.set_page_config(page_title="RX Diagnóstico IA", page_icon="🩻", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg:#050a0f; --surface:#0a1520; --border:#0d2035;
    --accent:#00d4ff; --ok:#00ff88; --warn:#ff6b35; --danger:#ff3355;
    --text:#c8d8e8; --muted:#4a6a8a;
}
* { font-family:'IBM Plex Sans',sans-serif !important; }
html,body,.stApp { background:var(--bg) !important; color:var(--text) !important; }
.block-container { padding:2rem 3rem !important; max-width:1400px !important; }
.rx-header { display:flex; align-items:center; gap:1.5rem; padding:1.5rem 2rem;
    background:var(--surface); border:1px solid var(--border); border-left:4px solid var(--accent);
    margin-bottom:2rem; border-radius:2px; }
.rx-header h1 { font-family:'IBM Plex Mono',monospace !important; font-size:1.4rem !important;
    font-weight:600 !important; color:var(--accent) !important; margin:0 !important; letter-spacing:2px; }
.rx-header p { font-size:0.8rem; color:var(--muted); margin:0.2rem 0 0 0;
    font-family:'IBM Plex Mono',monospace !important; letter-spacing:1px; }
.result-card { background:var(--surface); border:1px solid var(--border);
    border-radius:2px; padding:1.5rem; margin-bottom:1rem; }
.result-card h3 { font-family:'IBM Plex Mono',monospace !important; color:var(--accent) !important;
    font-size:0.85rem !important; letter-spacing:2px; margin-bottom:1.2rem !important;
    border-bottom:1px solid var(--border); padding-bottom:0.8rem; }
.patho-row { display:flex; align-items:center; margin-bottom:0.7rem; gap:0.8rem; }
.patho-name { font-family:'IBM Plex Mono',monospace !important; font-size:0.78rem;
    color:var(--text); width:200px; flex-shrink:0; }
.patho-bar-bg { flex:1; height:6px; background:#0d2035; border-radius:3px; overflow:hidden; }
.patho-bar-fill { height:100%; border-radius:3px; }
.patho-pct { font-family:'IBM Plex Mono',monospace !important; font-size:0.78rem;
    width:50px; text-align:right; flex-shrink:0; }
.badge-normal { display:inline-block; padding:0.3rem 0.8rem; border-radius:2px;
    font-family:'IBM Plex Mono',monospace !important; font-size:0.75rem; letter-spacing:1px;
    font-weight:600; background:rgba(0,255,136,0.1); color:var(--ok); border:1px solid rgba(0,255,136,0.3); }
.badge-warn { display:inline-block; padding:0.3rem 0.8rem; border-radius:2px;
    font-family:'IBM Plex Mono',monospace !important; font-size:0.75rem; letter-spacing:1px;
    font-weight:600; background:rgba(255,107,53,0.1); color:var(--warn); border:1px solid rgba(255,107,53,0.3); }
.badge-danger { display:inline-block; padding:0.3rem 0.8rem; border-radius:2px;
    font-family:'IBM Plex Mono',monospace !important; font-size:0.75rem; letter-spacing:1px;
    font-weight:600; background:rgba(255,51,85,0.1); color:var(--danger); border:1px solid rgba(255,51,85,0.3); }
.disclaimer { background:rgba(255,107,53,0.05); border:1px solid rgba(255,107,53,0.2);
    border-radius:2px; padding:1rem 1.5rem; margin-top:2rem; font-size:0.75rem;
    color:var(--muted); font-family:'IBM Plex Mono',monospace !important; }
#MainMenu,footer,header { visibility:hidden; }
.stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="rx-header">
    <div style="font-size:2.5rem">🩻</div>
    <div>
        <h1>RX DIAGNÓSTICO IA v5.0</h1>
        <p>SISTEMA DE APOYO AL DIAGNÓSTICO · CANARIAS SCS · SOLO USO INVESTIGACIÓN</p>
    </div>
</div>
""", unsafe_allow_html=True)

HF_MODEL_URL = "https://huggingface.co/LASIELL/rx-modelo/resolve/main/densenet_finetuned.pth"
MODEL_CACHE  = "/tmp/densenet_finetuned.pth"

@st.cache_resource
def cargar_modelo():
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    modelo.op_threshs = None
    num_ftrs = modelo.classifier.in_features
    modelo.classifier = nn.Linear(num_ftrs, 14)

    path_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos_entrenados", "densenet_finetuned.pth")
    if os.path.exists(path_local):
        path_modelo = path_local
    elif os.path.exists(MODEL_CACHE):
        path_modelo = MODEL_CACHE
    else:
        try:
            import urllib.request
            urllib.request.urlretrieve(HF_MODEL_URL, MODEL_CACHE)
            path_modelo = MODEL_CACHE
        except Exception as e:
            path_modelo = None

    if path_modelo:
        try:
            modelo.load_state_dict(torch.load(path_modelo, map_location="cpu", weights_only=True), strict=False)
        except:
            try:
                modelo.load_state_dict(torch.load(path_modelo, map_location="cpu"), strict=False)
            except:
                pass

    modelo.eval()
    return modelo

def analizar(imagen_pil, modelo):
    img = imagen_pil.convert("L")
    img_np = np.array(img).astype(np.float32)
    img_np = xrv.datasets.normalize(img_np, 255)
    img_t = torch.from_numpy(img_np[None, None, :, :]).float()
    img_t = torch.nn.functional.interpolate(img_t, size=(224, 224))
    with torch.no_grad():
        feats = modelo.features2(img_t)
        preds = torch.sigmoid(modelo.classifier(feats)).cpu().numpy()[0]
    return preds

def get_color(pct):
    if pct >= 60: return "#ff3355"
    if pct >= 35: return "#ff6b35"
    if pct >= 15: return "#ffaa00"
    return "#1a3a5a"

def get_badge(max_pct):
    if max_pct >= 60:
        return '<span class="badge-danger">⚠ HALLAZGO SIGNIFICATIVO</span>'
    if max_pct >= 30:
        return '<span class="badge-warn">◈ HALLAZGO MENOR</span>'
    return '<span class="badge-normal">✓ SIN HALLAZGOS RELEVANTES</span>'

col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.markdown('<div class="result-card"><h3>📂 CARGAR RADIOGRAFÍA</h3>', unsafe_allow_html=True)
    archivo = st.file_uploader("Selecciona una imagen RX", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if archivo:
        img = Image.open(archivo)
        st.image(img, use_container_width=True, caption=archivo.name)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#1a3a5a;
             border:2px dashed #0d2035;border-radius:4px;margin-top:1rem;">
            <div style="font-size:3rem">🩻</div>
            <p style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;margin-top:1rem;">
                ARRASTRA UNA RADIOGRAFÍA AQUÍ<br>O USA EL BOTÓN DE ARRIBA
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_der:
    st.markdown('<div class="result-card"><h3>🔬 ANÁLISIS IA</h3>', unsafe_allow_html=True)
    if archivo:
        with st.spinner("Analizando..."):
            modelo = cargar_modelo()
            img = Image.open(archivo)
            preds = analizar(img, modelo)

        resultados = sorted(zip(CATEGORIAS, preds), key=lambda x: x[1], reverse=True)
        max_pct = resultados[0][1] * 100

        st.markdown(get_badge(max_pct), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        barras_html = ""
        for cat, prob in resultados:
            pct = prob * 100
            color = get_color(pct)
            barras_html += f"""
            <div class="patho-row">
                <span class="patho-name">{cat}</span>
                <div class="patho-bar-bg">
                    <div class="patho-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
                </div>
                <span class="patho-pct" style="color:{color};">{pct:.1f}%</span>
            </div>
            """
        st.markdown(barras_html, unsafe_allow_html=True)

        top_cat, top_prob = resultados[0]
        st.markdown(f"""
        <div style="margin-top:1.5rem;padding:1rem;background:#050a0f;border-left:3px solid #00d4ff;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#4a6a8a;">DIAGNÓSTICO PRINCIPAL</span><br>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:#00d4ff;font-weight:600;">
                {top_cat.upper()}
            </span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:#c8d8e8;">
                &nbsp;{top_prob*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;color:#1a3a5a;">
            <p style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;">
                ESPERANDO IMAGEN...
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠ AVISO LEGAL: Este sistema es una herramienta de investigación en desarrollo.
    No está certificado como dispositivo médico (MDR 2017/745).
    Todos los resultados deben ser verificados por un médico cualificado.
    No utilizar como único criterio diagnóstico.
</div>
""", unsafe_allow_html=True)
