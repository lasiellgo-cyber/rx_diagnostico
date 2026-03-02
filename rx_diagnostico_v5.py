"""
RX DIAGNÓSTICO IA v5.0 - CONSULTA CANARIAS
"""
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchxrayvision as xrv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "modelos_entrenados", "densenet_finetuned.pth")

CATEGORIAS = [
    "Atelectasia", "Cardiomegalia", "Efusión", "Infiltración", "Masa",
    "Nódulo", "Neumonía", "Neumotórax", "Consolidación", "Edema",
    "Enfisema", "Fibrosis", "Engrosamiento Pleural", "Hernia"
]

ZONA_ANATOMICA = {
    "Atelectasia":           "Pulmón (lóbulo inferior)",
    "Cardiomegalia":         "Corazón / Mediastino",
    "Efusión":               "Espacio pleural",
    "Infiltración":          "Parénquima pulmonar",
    "Masa":                  "Pulmón / Mediastino",
    "Nódulo":                "Parénquima pulmonar",
    "Neumonía":              "Pulmón (consolidación alveolar)",
    "Neumotórax":            "Espacio pleural / Apex pulmonar",
    "Consolidación":         "Parénquima pulmonar",
    "Edema":                 "Pulmón bilateral / Hilio",
    "Enfisema":              "Pulmón (hiperinsuflación)",
    "Fibrosis":              "Intersticio pulmonar",
    "Engrosamiento Pleural": "Pleura",
    "Hernia":                "Diafragma / Mediastino inferior",
}

HALLAZGO_VISUAL = {
    "Atelectasia":           "Opacidad laminar con pérdida de volumen y desplazamiento de fisuras",
    "Cardiomegalia":         "Índice cardiotorácico > 0.5, silueta cardíaca aumentada",
    "Efusión":               "Opacificación del seno costofrénico, menisco pleural",
    "Infiltración":          "Opacidades heterogéneas de predominio peribronquial",
    "Masa":                  "Opacidad redondeada > 3 cm con bordes bien definidos",
    "Nódulo":                "Opacidad redondeada < 3 cm, puede tener bordes espiculados",
    "Neumonía":              "Consolidación alveolar con broncograma aéreo",
    "Neumotórax":            "Línea pleural visible, ausencia de trama vascular periférica",
    "Consolidación":         "Opacidad homogénea con broncograma aéreo positivo",
    "Edema":                 "Opacidades bilaterales perihiliares en alas de mariposa, líneas B de Kerley",
    "Enfisema":              "Hiperinsuflación, aplanamiento diafragmático, aumento espacio intercostal",
    "Fibrosis":              "Opacidades reticulares basales bilaterales, patrón en panal de abeja",
    "Engrosamiento Pleural": "Opacidad pleural periférica sin menisco, bordes irregulares",
    "Hernia":                "Estructura abdominal por encima del diafragma, nivel hidroaéreo",
}

UMBRAL_POSITIVO = 0.45
UMBRAL_SUGESTIVO = 0.25

st.set_page_config(
    page_title="RX Diagnóstico IA — Canarias",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg:#05080f; --surface:#0a0f1a; --card:#0f1726; --border:#1a2540;
    --accent:#00d4ff; --green:#00e676; --red:#ff3d5a; --yellow:#ffc400;
    --text:#c8d8f0; --muted:#4a6080;
    --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
}
* { font-family: var(--sans) !important; }
.stApp { background: var(--bg) !important; color: var(--text) !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1300px !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

.rx-header { display:flex; align-items:center; gap:16px; padding:20px 28px;
    background:var(--card); border:1px solid var(--border); border-radius:12px; margin-bottom:24px; }
.rx-header-icon { font-size:2.5rem; }
.rx-header-title { font-family:var(--mono) !important; font-size:1.4rem; font-weight:600; color:var(--accent); letter-spacing:2px; }
.rx-header-sub { font-size:0.8rem; color:var(--muted); letter-spacing:1px; margin-top:2px; }
.rx-badge { margin-left:auto; background:#001a30; border:1px solid var(--accent); color:var(--accent);
    font-family:var(--mono) !important; font-size:0.7rem; padding:4px 12px; border-radius:4px; letter-spacing:1px; }

.result-normal { background:linear-gradient(135deg,#001a0a,#002210); border:2px solid var(--green);
    border-radius:12px; padding:24px 28px; margin:16px 0; }
.result-normal .label { font-family:var(--mono) !important; font-size:0.75rem; color:var(--green); letter-spacing:3px; margin-bottom:8px; }
.result-normal .value { font-size:2rem; font-weight:600; color:var(--green); }

.result-anormal { background:linear-gradient(135deg,#1a0008,#220010); border:2px solid var(--red);
    border-radius:12px; padding:24px 28px; margin:16px 0; }
.result-anormal .label { font-family:var(--mono) !important; font-size:0.75rem; color:var(--red); letter-spacing:3px; margin-bottom:8px; }
.result-anormal .value { font-size:2rem; font-weight:600; color:var(--red); }

.diag-section { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px 24px; margin:12px 0; }
.diag-num { font-family:var(--mono) !important; font-size:0.7rem; color:var(--accent); letter-spacing:2px; margin-bottom:6px; }
.diag-title { font-size:1rem; font-weight:600; color:var(--text); margin-bottom:14px; }

.zona-tag { display:inline-block; background:#0a1530; border:1px solid var(--border); color:var(--accent);
    font-family:var(--mono) !important; font-size:0.75rem; padding:4px 10px; border-radius:4px; margin:4px 4px 4px 0; }

.hallazgo-item { border-left:3px solid var(--yellow); padding:8px 14px; margin:8px 0;
    background:#0d1220; border-radius:0 6px 6px 0; }
.hallazgo-name { font-family:var(--mono) !important; font-size:0.8rem; color:var(--yellow); margin-bottom:3px; }
.hallazgo-desc { font-size:0.85rem; color:var(--text); }

.diag-final { background:linear-gradient(135deg,#0a0f20,#0f1530); border:1px solid var(--accent);
    border-radius:10px; padding:20px 24px; margin:12px 0; }
.diag-final-title { font-family:var(--mono) !important; font-size:0.7rem; color:var(--accent); letter-spacing:3px; margin-bottom:12px; }
.diag-final-item { font-size:1rem; color:white; font-weight:500; padding:6px 0; border-bottom:1px solid var(--border); }
.diag-final-item:last-child { border-bottom:none; }

.prob-row { display:flex; align-items:center; gap:12px; margin:8px 0; }
.prob-name { font-family:var(--mono) !important; font-size:0.85rem; color:var(--text); width:200px; flex-shrink:0; }
.prob-bar-bg { flex:1; height:8px; background:#0a1020; border-radius:4px; overflow:hidden; }
.prob-pct { font-family:var(--mono) !important; font-size:0.8rem; width:45px; text-align:right; }

.disclaimer { background:#0a0c10; border:1px solid #1a1f2a; border-radius:8px; padding:12px 16px;
    margin-top:20px; font-size:0.75rem; color:var(--muted); text-align:center; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    modelo.op_threshs = None
    num_ftrs = modelo.classifier.in_features
    modelo.classifier = nn.Linear(num_ftrs, 14)
    tipo = "BASE"
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            modelo.load_state_dict(state, strict=False)
            tipo = "ENTRENADO"
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

# ── HEADER ──
st.markdown("""
<div class="rx-header">
    <div class="rx-header-icon">🩻</div>
    <div>
        <div class="rx-header-title">RX DIAGNÓSTICO IA</div>
        <div class="rx-header-sub">SISTEMA DE APOYO AL DIAGNÓSTICO · CANARIAS SCS</div>
    </div>
    <div class="rx-badge">v5.0 · SOLO USO INVESTIGACIÓN</div>
</div>
""", unsafe_allow_html=True)

modelo, tipo_modelo = cargar_modelo()

col_izq, col_der = st.columns([1, 1.4], gap="large")

with col_izq:
    st.markdown("##### Subir radiografía")
    archivo = st.file_uploader("Imagen", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if archivo:
        imagen = Image.open(archivo)
        st.image(imagen, use_container_width=True)
        st.markdown(f'<div style="font-family:monospace;font-size:0.75rem;color:#4a6080;text-align:center;margin-top:6px;">{archivo.name} · {tipo_modelo}</div>', unsafe_allow_html=True)

with col_der:
    if archivo:
        with st.spinner("Analizando..."):
            preds = analizar(Image.open(archivo), modelo)

        positivos  = sorted([(CATEGORIAS[i], preds[i]) for i in range(14) if preds[i] >= UMBRAL_POSITIVO], key=lambda x: -x[1])
        sugestivos = sorted([(CATEGORIAS[i], preds[i]) for i in range(14) if UMBRAL_SUGESTIVO <= preds[i] < UMBRAL_POSITIVO], key=lambda x: -x[1])
        todas = positivos + sugestivos
        es_anormal = len(positivos) > 0

        # PASO 1: NORMAL / ANORMAL
        if es_anormal:
            st.markdown(f"""<div class="result-anormal">
                <div class="label">▸ RESULTADO GLOBAL</div>
                <div class="value">⚠ ANORMAL</div>
                <div style="font-size:0.85rem;color:#ff8095;margin-top:8px;">{len(positivos)} hallazgo{'s' if len(positivos)>1 else ''} detectado{'s' if len(positivos)>1 else ''}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-normal">
                <div class="label">▸ RESULTADO GLOBAL</div>
                <div class="value">✓ NORMAL</div>
                <div style="font-size:0.85rem;color:#80e8a0;margin-top:8px;">No se detectan hallazgos patológicos significativos</div>
            </div>""", unsafe_allow_html=True)

        if todas:
            # PASO 2: ZONA ANATÓMICA
            zonas = list(dict.fromkeys([ZONA_ANATOMICA[p[0]] for p in todas]))
            zonas_html = "".join([f'<span class="zona-tag">{z}</span>' for z in zonas])
            st.markdown(f"""<div class="diag-section">
                <div class="diag-num">▸ PASO 2 / 4</div>
                <div class="diag-title">Zona anatómica afectada</div>
                {zonas_html}
            </div>""", unsafe_allow_html=True)

            # PASO 3: QUÉ SE VE
            hallazgos_html = "".join([f"""<div class="hallazgo-item">
                <div class="hallazgo-name">{cat} — {prob*100:.0f}%</div>
                <div class="hallazgo-desc">{HALLAZGO_VISUAL[cat]}</div>
            </div>""" for cat, prob in todas])
            st.markdown(f"""<div class="diag-section">
                <div class="diag-num">▸ PASO 3 / 4</div>
                <div class="diag-title">Hallazgos radiológicos</div>
                {hallazgos_html}
            </div>""", unsafe_allow_html=True)

            # PASO 4: DIAGNÓSTICO FINAL
            if positivos:
                items_html = "".join([f'<div class="diag-final-item">{"🔴" if i==0 else "🟡"} {cat} <span style="color:#4a6080;font-size:0.8rem;">({prob*100:.0f}%)</span></div>' 
                                      for i, (cat, prob) in enumerate(positivos[:3])])
            else:
                items_html = '<div class="diag-final-item">⚪ Hallazgos sugestivos — correlacionar clínicamente</div>'
            
            st.markdown(f"""<div class="diag-final">
                <div class="diag-final-title">▸ PASO 4 / 4 · DIAGNÓSTICO PRINCIPAL</div>
                {items_html}
            </div>""", unsafe_allow_html=True)

            with st.expander("Ver todas las probabilidades"):
                todas_ord = sorted(zip(CATEGORIAS, preds), key=lambda x: -x[1])
                bars = ""
                for cat, prob in todas_ord:
                    w = max(int(prob*100), 1)
                    col = "#ff3d5a" if prob >= UMBRAL_POSITIVO else ("#ffc400" if prob >= UMBRAL_SUGESTIVO else "#1a3a5a")
                    tc = "#ff3d5a" if prob >= UMBRAL_POSITIVO else ("#ffc400" if prob >= UMBRAL_SUGESTIVO else "#4a6080")
                    bars += f"""<div class="prob-row">
                        <div class="prob-name">{cat}</div>
                        <div class="prob-bar-bg"><div style="height:100%;width:{w}%;border-radius:4px;background:{col};"></div></div>
                        <div class="prob-pct" style="color:{tc};">{prob*100:.0f}%</div>
                    </div>"""
                st.markdown(bars, unsafe_allow_html=True)
    else:
        st.markdown("""<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            height:400px;color:#2a4060;text-align:center;">
            <div style="font-size:4rem;margin-bottom:16px;">🩻</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;letter-spacing:2px;">ESPERANDO IMAGEN</div>
            <div style="font-size:0.8rem;margin-top:8px;color:#1a3050;">Sube una radiografía para analizar</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""<div class="disclaimer">
    ⚕️ HERRAMIENTA DE APOYO AL DIAGNÓSTICO — No reemplaza el criterio clínico ni el informe radiológico. 
    Solo para uso en investigación. Correlacionar siempre con la clínica del paciente.
</div>""", unsafe_allow_html=True)
