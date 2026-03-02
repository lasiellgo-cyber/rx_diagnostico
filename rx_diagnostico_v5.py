"""
RX DIAGNÓSTICO IA v5.0 - INTERFAZ MÉDICA STREAMLIT
Analiza automáticamente al subir la imagen.
Diagnóstico: NORMAL / ANORMAL + patologías + conclusión.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchxrayvision as xrv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "modelos_entrenados", "densenet_finetuned.pth")

CATEGORIAS = [
    "Atelectasia", "Cardiomegalia", "Efusión Pleural", "Infiltración", "Masa",
    "Nódulo", "Neumonía", "Neumotórax", "Consolidación", "Edema",
    "Enfisema", "Fibrosis", "Engrosamiento Pleural", "Hernia"
]

UMBRAL_POSITIVO = 0.45
UMBRAL_NORMAL   = 0.20

st.set_page_config(page_title="RX Diagnóstico IA", page_icon="🩻", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
:root { --bg:#070d14; --surface:#0d1825; --border:#1a2e45; --accent:#00d4ff; --green:#00e676; --red:#ff3d5a; --yellow:#ffcc00; --text:#c8d8e8; --muted:#4a6a8a; }
* { font-family:'IBM Plex Sans',sans-serif; }
.stApp { background:var(--bg); color:var(--text); }
.block-container { padding:2rem 3rem; max-width:1200px; }
h1,h2,h3,h4 { font-family:'IBM Plex Mono',monospace !important; color:var(--accent) !important; }
.header-box { border:1px solid var(--border); background:var(--surface); padding:1.5rem 2rem; margin-bottom:2rem; border-left:4px solid var(--accent); }
.header-title { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; color:var(--accent); letter-spacing:3px; margin:0; }
.header-sub { color:var(--muted); font-size:0.85rem; margin-top:0.3rem; letter-spacing:1px; }
.result-normal { background:rgba(0,230,118,0.08); border:2px solid var(--green); border-radius:4px; padding:1.5rem 2rem; margin:1rem 0; }
.result-anormal { background:rgba(255,61,90,0.08); border:2px solid var(--red); border-radius:4px; padding:1.5rem 2rem; margin:1rem 0; }
.veredicto-normal { font-family:'IBM Plex Mono',monospace; font-size:2rem; font-weight:600; color:var(--green); letter-spacing:4px; }
.veredicto-anormal { font-family:'IBM Plex Mono',monospace; font-size:2rem; font-weight:600; color:var(--red); letter-spacing:4px; }
.patologia-row { margin:0.4rem 0; padding:0.5rem 0.8rem; background:rgba(255,255,255,0.02); border-radius:3px; }
.diagnostico-box { background:var(--surface); border:1px solid var(--border); border-top:3px solid var(--accent); padding:1.5rem; margin-top:1.5rem; }
.diagnostico-titulo { font-family:'IBM Plex Mono',monospace; color:var(--accent); font-size:0.8rem; letter-spacing:2px; margin-bottom:0.8rem; }
.aviso { background:rgba(255,204,0,0.05); border-left:3px solid var(--yellow); padding:0.8rem 1rem; margin-top:2rem; font-size:0.8rem; color:var(--muted); }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-box"><p class="header-title">🩻 RX DIAGNÓSTICO IA</p><p class="header-sub">SISTEMA DE APOYO AL DIAGNÓSTICO RADIOLÓGICO · SCS · v5.0</p></div>', unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    modelo.op_threshs = None
    num_ftrs = modelo.classifier.in_features
    modelo.classifier = nn.Linear(num_ftrs, 14)
    if os.path.exists(PATH_MODELO):
        try:
            modelo.load_state_dict(torch.load(PATH_MODELO, map_location="cpu"), strict=False)
        except:
            pass
    modelo.to(DEVICE).eval()
    return modelo

col_img, col_resultado = st.columns([1, 1], gap="large")

with col_img:
    st.markdown("#### 📤 SUBIR RADIOGRAFÍA")
    archivo = st.file_uploader("Arrastra o selecciona la imagen", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if archivo:
        st.image(Image.open(archivo), use_container_width=True, caption=archivo.name)

with col_resultado:
    if archivo is None:
        st.markdown('<div style="height:200px;display:flex;align-items:center;justify-content:center;border:1px dashed #1a2e45;background:#0d1825;margin-top:2rem;"><p style="color:#4a6a8a;font-family:IBM Plex Mono,monospace;font-size:0.9rem;letter-spacing:2px;">ESPERANDO IMAGEN...</p></div>', unsafe_allow_html=True)
    else:
        with st.spinner("🔬 Analizando radiografía..."):
            modelo = cargar_modelo()
            img_ia = Image.open(archivo).convert("L")
            img_arr = xrv.datasets.normalize(np.array(img_ia), 255)
            img_tensor = torch.from_numpy(img_arr[None, None]).float().to(DEVICE)
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=(224, 224))
            with torch.no_grad():
                feats = modelo.features2(img_tensor)
                preds = torch.sigmoid(modelo.classifier(feats)).cpu().numpy()[0]

        resultados = sorted(zip(CATEGORIAS, preds), key=lambda x: x[1], reverse=True)
        hallazgos_positivos = [(c, p) for c, p in resultados if p >= UMBRAL_POSITIVO]
        hallazgos_leves     = [(c, p) for c, p in resultados if UMBRAL_NORMAL <= p < UMBRAL_POSITIVO]
        es_normal = len(hallazgos_positivos) == 0

        if es_normal:
            st.markdown('<div class="result-normal"><div class="veredicto-normal">✓ NORMAL</div><p style="margin:0.5rem 0 0;color:#a0c8a0;">No se detectan hallazgos patológicos significativos.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-anormal"><div class="veredicto-anormal">⚠ ANORMAL</div><p style="margin:0.5rem 0 0;color:#ffa0a0;">Se detectan hallazgos que requieren valoración clínica.</p></div>', unsafe_allow_html=True)

        st.markdown("#### 📊 ANÁLISIS DETALLADO")
        for cat, prob in resultados[:10]:
            pct = prob * 100
            if prob >= UMBRAL_POSITIVO:
                color = "#ff3d5a"; icono = "🔴"
            elif prob >= UMBRAL_NORMAL:
                color = "#ffcc00"; icono = "🟡"
            else:
                color = "#4a6a8a"; icono = "⚪"
            ancho = min(int(prob * 100), 100)
            st.markdown(f'<div class="patologia-row"><span style="font-family:IBM Plex Mono,monospace;font-size:0.9rem;">{icono} {cat}</span><span style="float:right;font-family:IBM Plex Mono,monospace;font-size:0.85rem;color:{color};">{pct:.1f}%</span><div style="background:rgba(255,255,255,0.05);border-radius:2px;height:6px;margin-top:4px;"><div style="background:{color};width:{ancho}%;height:6px;border-radius:2px;"></div></div></div>', unsafe_allow_html=True)

        st.markdown('<div class="diagnostico-box"><p class="diagnostico-titulo">CONCLUSIÓN DIAGNÓSTICA</p>', unsafe_allow_html=True)
        if es_normal:
            st.markdown("**Radiografía dentro de límites normales.** No se identifican opacidades, consolidaciones, derrames ni otras alteraciones significativas.")
        else:
            items = [f"**{c}** ({p*100:.0f}%)" for c, p in hallazgos_positivos]
            if hallazgos_leves:
                leves = [f"{c} ({p*100:.0f}%)" for c, p in hallazgos_leves[:3]]
                items.append(f"Hallazgos leves a valorar: {', '.join(leves)}")
            st.markdown(f"Se identifican: {', '.join(items)}. **Se recomienda correlación clínica y valoración por especialista.**")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="aviso">⚠️ SISTEMA DE APOYO AL DIAGNÓSTICO — No reemplaza el juicio clínico del médico. Uso exclusivo como herramienta de ayuda.</div>', unsafe_allow_html=True)
