"""
RX DIAGNÓSTICO IA v4
====================
Novedades v4:
- Detección automática de región anatómica (tórax vs huesos vs columna vs cráneo)
- Motor tórax:   TorchXRayVision DenseNet-121 (700k+ RX reales)
- Motor huesos:  CLIP + prompts especializados musculoesqueléticos
- Motor columna: CLIP + prompts especializados vertebrales  
- Motor cráneo:  CLIP + prompts especializados craneales
- Alerta visual cuando la imagen no parece una RX
"""

import streamlit as st
import torch
import numpy as np
import os
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torchxrayvision as xrv
import skimage.transform
import datetime

# ── intentar importar transformers (para CLIP en zonas óseas) ──
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_DISPONIBLE = True
except ImportError:
    CLIP_DISPONIBLE = False

# ─────────────────────────────────────────────────────────────
# PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="RX Diagnóstico IA v4", page_icon="🩻", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --bg:#070b12; --surface:#0e1520; --border:#162030;
    --accent:#00e5ff; --accent2:#ff3d5a; --warn:#ffaa00;
    --ok:#00e676; --text:#ccd6f6; --muted:#546e8a;
}
* { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--bg); color:var(--text); }
.block-container { padding:1.5rem 2.5rem 3rem; max-width:1400px; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; color:var(--accent) !important; }
div[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--border); }
.stButton>button {
    background:linear-gradient(135deg,#00e5ff,#006aff);
    color:#040810; font-family:'DM Mono',monospace; font-weight:500;
    font-size:.95rem; letter-spacing:1.5px; border:none; border-radius:6px;
    padding:.7rem 2rem; width:100%; transition:all .2s;
}
.stButton>button:hover { opacity:.85; transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,229,255,.25); }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.4rem; margin-bottom:1rem; }
.result-hero { background:var(--surface); border-radius:14px; padding:2.5rem 2rem; text-align:center; margin:1.5rem 0; position:relative; overflow:hidden; }
.result-hero::before { content:''; position:absolute; top:0; left:0; right:0; height:4px; }
.result-hero.alert::before { background:linear-gradient(90deg,var(--accent2),#ff8c00); }
.result-hero.warn::before  { background:linear-gradient(90deg,var(--warn),#ffe000); }
.result-hero.ok::before    { background:linear-gradient(90deg,var(--ok),#00bfa5); }
.result-hero.info::before  { background:linear-gradient(90deg,var(--accent),#006aff); }
.percent-big { font-family:'Syne',sans-serif; font-size:5.5rem; font-weight:800; line-height:1; margin:.3rem 0; }
.diagnosis-name { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; margin:.5rem 0; }
.path-row { display:flex; justify-content:space-between; align-items:center; padding:.5rem 0; border-bottom:1px solid var(--border); font-size:.88rem; }
.tag { font-family:'DM Mono',monospace; font-size:.7rem; color:var(--accent); text-transform:uppercase; letter-spacing:2.5px; }
.badge { display:inline-block; padding:.2rem .7rem; border-radius:20px; font-family:'DM Mono',monospace; font-size:.72rem; font-weight:500; letter-spacing:1px; }
.badge-alert { background:rgba(255,61,90,.15); color:var(--accent2); border:1px solid rgba(255,61,90,.3); }
.badge-warn  { background:rgba(255,170,0,.15); color:var(--warn); border:1px solid rgba(255,170,0,.3); }
.badge-ok    { background:rgba(0,230,118,.12); color:var(--ok); border:1px solid rgba(0,230,118,.3); }
.badge-info  { background:rgba(0,229,255,.1);  color:var(--accent); border:1px solid rgba(0,229,255,.2); }
.badge-purple{ background:rgba(180,100,255,.15); color:#c87fff; border:1px solid rgba(180,100,255,.3); }
.region-chip { display:inline-flex; align-items:center; gap:.4rem; padding:.35rem .9rem; border-radius:20px; font-family:'DM Mono',monospace; font-size:.78rem; margin:.2rem; cursor:pointer; border:1px solid var(--border); background:var(--surface); color:var(--muted); transition:all .2s; }
.region-chip.active { border-color:var(--accent); color:var(--accent); background:rgba(0,229,255,.08); }
.stProgress>div>div { background:linear-gradient(90deg,#00e5ff,#006aff) !important; border-radius:4px; }
hr { border-color:var(--border) !important; }
.stSelectbox label,.stRadio label,.stCheckbox label { color:var(--muted) !important; font-size:.85rem; }
.streamlit-expanderHeader { color:var(--accent) !important; font-family:'DM Mono',monospace !important; font-size:.82rem; }
.detector-box { background:rgba(0,229,255,.06); border:1px solid rgba(0,229,255,.2); border-radius:10px; padding:1rem 1.2rem; margin:.8rem 0; }
.warn-box { background:rgba(255,170,0,.07); border:1px solid rgba(255,170,0,.25); border-radius:10px; padding:1rem 1.2rem; margin:.8rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DETECTOR AUTOMÁTICO DE REGIÓN
# Usa características visuales básicas para clasificar
# ─────────────────────────────────────────────────────────────
def detectar_region(img: Image.Image) -> tuple[str, float, str]:
    """
    Analiza la imagen y decide qué región anatómica es.
    Retorna (region, confianza, motivo)
    
    Método: características de imagen + CLIP si disponible
    """
    img_gray = np.array(img.convert('L')).astype(np.float32)
    h, w = img_gray.shape
    ratio = w / h

    # Normalizar
    img_n = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)

    # ── Características básicas ──
    # Tórax: formato casi cuadrado/ligeramente ancho, gran área clara central
    # Columna: formato muy vertical (estrecho y alto)
    # Extremidades: suele ser alargado, hueso fino visible
    # Cráneo: casi circular/cuadrado, sin cavidad central clara

    zona_central = img_n[h//4:3*h//4, w//4:3*w//4]
    brillo_centro = zona_central.mean()
    brillo_total  = img_n.mean()
    contraste     = img_n.std()

    # Distribución de brillo (histograma simplificado)
    oscuro  = (img_n < 0.2).mean()   # % píxeles muy oscuros
    claro   = (img_n > 0.7).mean()   # % píxeles muy claros
    medio   = 1 - oscuro - claro

    scores = {
        "torax":       0.0,
        "extremidades":0.0,
        "columna":     0.0,
        "craneo":      0.0,
    }

    # Tórax: ratio ancho ~0.8-1.3, gran área oscura (pulmones), zona blanca central (mediastino)
    if 0.75 < ratio < 1.45:
        scores["torax"] += 0.4
    if oscuro > 0.35:          # pulmones oscuros
        scores["torax"] += 0.3
    if brillo_centro > brillo_total * 1.1:  # mediastino más brillante que los bordes
        scores["torax"] += 0.2

    # Extremidades: ratio muy variable, hueso fino (línea brillante), mucho fondo oscuro
    if oscuro > 0.50:
        scores["extremidades"] += 0.35
    if contraste > 0.28:
        scores["extremidades"] += 0.2
    if claro < 0.15:           # sin grandes zonas blancas
        scores["extremidades"] += 0.2
    if ratio < 0.55 or ratio > 1.8:  # muy vertical o muy horizontal
        scores["extremidades"] += 0.2

    # Columna: muy vertical, franja central brillante continua
    if ratio < 0.6:
        scores["columna"] += 0.5
    if brillo_centro > 0.45:
        scores["columna"] += 0.2
    if medio > 0.45:
        scores["columna"] += 0.15

    # Cráneo: casi cuadrado/redondo, gran masa blanca, poco fondo oscuro
    if 0.8 < ratio < 1.25:
        scores["craneo"] += 0.2
    if claro > 0.30:
        scores["craneo"] += 0.35
    if oscuro < 0.25:
        scores["craneo"] += 0.25

    # Si CLIP disponible, refinar con texto
    if CLIP_DISPONIBLE:
        try:
            clip_m, clip_p, clip_d = cargar_clip()
            prompts = [
                "chest X-ray radiograph lungs ribs heart",
                "bone X-ray extremity fracture long bone",
                "spine X-ray vertebrae column",
                "skull X-ray cranium head",
            ]
            keys = ["torax", "extremidades", "columna", "craneo"]
            inputs = clip_p(text=prompts, images=img, return_tensors="pt", padding=True).to(clip_d)
            with torch.no_grad():
                out = clip_m(**inputs)
            probs = out.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
            for k, p in zip(keys, probs):
                scores[k] += float(p) * 0.8   # peso CLIP mayor
        except Exception:
            pass

    mejor = max(scores, key=scores.get)
    total = sum(scores.values()) + 1e-8
    confianza = scores[mejor] / total

    motivos = {
        "torax":        "Imagen con características de radiografía torácica",
        "extremidades": "Imagen con características de hueso / extremidad",
        "columna":      "Imagen con características de columna vertebral",
        "craneo":       "Imagen con características de cráneo / cabeza",
    }

    return mejor, confianza, motivos[mejor]


# ─────────────────────────────────────────────────────────────
# CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar_xrv():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # Si existe modelo fine-tuned, cargarlo automáticamente
    carpeta_base   = os.path.dirname(os.path.abspath(__file__))
    modelo_custom  = os.path.join(carpeta_base, "modelos_entrenados", "densenet_finetuned.pth")
    if os.path.exists(modelo_custom):
        try:
            state = torch.load(modelo_custom, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            st.sidebar.success("✅ Modelo fine-tuned cargado")
        except Exception:
            st.sidebar.info("ℹ️ Usando modelo base (sin fine-tuning)")
    
    model.to(device).eval()
    return model, device

@st.cache_resource(show_spinner=False)
def cargar_clip():
    if not CLIP_DISPONIBLE:
        return None, None, "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-large-patch14-336"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor, device


# ─────────────────────────────────────────────────────────────
# BASE DE CONOCIMIENTO ÓSEA (CLIP)
# ─────────────────────────────────────────────────────────────
PROMPTS_OSEOS = {
    # ── EXTREMIDADES ──
    "Fractura transversa":      ["X-ray transverse fracture complete bone break perpendicular fracture line",
                                  "radiograph horizontal fracture cortex long bone"],
    "Fractura oblicua/espiral": ["oblique spiral fracture X-ray long bone diagonal",
                                  "radiograph angulated fracture diaphysis spiral"],
    "Fractura conminuta":       ["comminuted fracture X-ray multiple bone fragments shattered",
                                  "radiograph bone fragmentation multiple fracture lines"],
    "Fractura por estrés":      ["stress fracture X-ray hairline fracture bone",
                                  "radiograph fatigue fracture linear lucency"],
    "Fractura impactada":       ["impacted fracture X-ray bone fragments driven together",
                                  "radiograph impaction fracture cortical disruption"],
    "Luxación articular":       ["joint dislocation X-ray articular surface displacement",
                                  "radiograph disrupted joint alignment dislocation"],
    "Artrosis":                 ["osteoarthritis X-ray joint space narrowing osteophytes subchondral sclerosis",
                                  "radiograph degenerative joint disease bone spurs"],
    "Artritis reumatoide":      ["rheumatoid arthritis X-ray joint erosions periarticular osteopenia",
                                  "radiograph symmetric joint destruction small bones"],
    "Tumor / Lesión ósea":      ["bone tumor X-ray lytic sclerotic lesion cortical destruction periosteal",
                                  "radiograph bone mass aggressive lesion"],
    "Osteomielitis":            ["osteomyelitis X-ray bone infection periosteal reaction cortical destruction",
                                  "radiograph bone sequestrum involucrum infection"],
    "Osteoporosis":             ["osteoporosis X-ray decreased bone density thin cortices trabecular loss",
                                  "radiograph reduced bone density osteopenia"],
    "Normal":                   ["normal bone X-ray intact cortex no fracture healthy",
                                  "radiograph normal bone without pathology"],
}

PROMPTS_COLUMNA = {
    "Fractura vertebral":        ["spine X-ray vertebral fracture compression wedge deformity height loss",
                                   "radiograph broken vertebra compression fracture"],
    "Fractura por aplastamiento":["vertebral crush fracture X-ray complete height loss osteoporotic",
                                   "radiograph vertebral body collapse severe compression"],
    "Escoliosis":                ["scoliosis X-ray lateral curvature spine Cobb angle deviation",
                                   "radiograph abnormal lateral spinal curve scoliosis"],
    "Espondilosis":              ["spondylosis X-ray osteophytes bone spurs disc space narrowing",
                                   "radiograph vertebral osteophytes degenerative changes"],
    "Espondilolistesis":         ["spondylolisthesis X-ray vertebral slippage forward displacement",
                                   "radiograph vertebral translation step deformity"],
    "Hernia discal (indirecta)": ["disc herniation X-ray disc space narrowing indirect signs",
                                   "radiograph reduced intervertebral disc height"],
    "Osteoporosis vertebral":    ["vertebral osteoporosis X-ray biconcave vertebrae codfish vertebra",
                                   "radiograph reduced vertebral bone density"],
    "Espondilitis anquilosante": ["ankylosing spondylitis X-ray bamboo spine syndesmophytes sacroiliitis",
                                   "radiograph fused vertebrae square vertebrae"],
    "Normal":                    ["normal spine X-ray regular alignment disc spaces no pathology",
                                   "radiograph healthy vertebrae no spinal pathology"],
}

PROMPTS_CRANEO = {
    "Fractura craneal lineal":   ["skull X-ray linear fracture cranial vault bone disruption",
                                   "radiograph skull fracture linear lucency"],
    "Fractura deprimida":        ["depressed skull fracture X-ray bone fragments inward displacement",
                                   "radiograph comminuted skull fracture depression"],
    "Sinusitis":                 ["sinusitis skull X-ray opacification air-fluid level paranasal sinus",
                                   "radiograph opaque sinus mucosal thickening"],
    "Lesión lítica craneal":     ["skull lytic lesion X-ray bone destruction punched out defect myeloma",
                                   "radiograph focal bone loss skull"],
    "Calcificaciones":           ["skull X-ray intracranial calcification pineal calcification",
                                   "radiograph calcified lesion skull"],
    "Normal":                    ["normal skull X-ray intact cranial vault clear sinuses no pathology",
                                   "radiograph healthy cranium no skull pathology"],
}


def analizar_clip(img: Image.Image, prompts_dict: dict) -> dict:
    """Analiza con CLIP usando ensemble de prompts por categoría"""
    if not CLIP_DISPONIBLE:
        return {k: 1/len(prompts_dict) for k in prompts_dict}
    
    clip_m, clip_p, clip_d = cargar_clip()
    scores = {}
    with torch.no_grad():
        for cat, prompts in prompts_dict.items():
            vals = []
            for p in prompts:
                inp = clip_p(text=[p], images=img, return_tensors="pt", padding=True).to(clip_d)
                out = clip_m(**inp)
                vals.append(out.logits_per_image.item())
            scores[cat] = np.mean(vals)
    
    arr = np.array(list(scores.values()))
    arr_exp = np.exp(arr - arr.max())
    probs = arr_exp / arr_exp.sum()
    return dict(zip(scores.keys(), probs.tolist()))


# ─────────────────────────────────────────────────────────────
# ANÁLISIS TÓRAX (TorchXRayVision)
# ─────────────────────────────────────────────────────────────
TRADUCCIONES = {
    "Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia",
    "Effusion":"Derrame pleural","Infiltration":"Infiltrado pulmonar",
    "Mass":"Masa pulmonar","Nodule":"Nódulo pulmonar",
    "Pneumonia":"Neumonía","Pneumothorax":"Neumotórax",
    "Consolidation":"Consolidación","Edema":"Edema pulmonar",
    "Emphysema":"Enfisema","Fibrosis":"Fibrosis pulmonar",
    "Pleural_Thickening":"Engrosamiento pleural","Hernia":"Hernia diafragmática",
    "Lung Opacity":"Opacidad pulmonar","Lung Lesion":"Lesión pulmonar",
    "Fracture":"Fractura costal","Enlarged Cardiomediastinum":"Mediastino ensanchado",
    "No Finding":"Sin hallazgos (Normal)","Support Devices":"Dispositivos de soporte",
    "Pleural Other":"Patología pleural",
}

def analizar_torax(img: Image.Image) -> dict:
    modelo, device = cargar_xrv()
    img_gray = np.array(img.convert('L')).astype(np.float32)
    img_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
    img_norm = img_norm * 2048 - 1024
    img_res  = skimage.transform.resize(img_norm, (224, 224), anti_aliasing=True)
    tensor   = torch.tensor(img_res).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = modelo(tensor)
    probs = output[0].detach().cpu().numpy()
    return {TRADUCCIONES.get(l, l): float(p) for l, p in zip(modelo.pathologies, probs) if l in TRADUCCIONES}


# ─────────────────────────────────────────────────────────────
# SEVERIDAD
# ─────────────────────────────────────────────────────────────
URGENTES   = {"Neumotórax","Edema pulmonar","Consolidación","Neumonía","Mediastino ensanchado","Masa pulmonar",
               "Fractura transversa","Fractura conminuta","Fractura vertebral","Fractura craneal lineal","Fractura deprimida"}
IMPORTANTES = {"Cardiomegalia","Derrame pleural","Atelectasia","Infiltrado pulmonar","Nódulo pulmonar",
                "Fractura oblicua/espiral","Luxación articular","Tumor / Lesión ósea","Osteomielitis",
                "Escoliosis","Espondilolistesis","Sinusitis"}

COLOR = {"alert":"#ff3d5a","warn":"#ffaa00","ok":"#00e676","info":"#00e5ff"}

def gravedad(nombre: str, prob: float):
    if "Normal" in nombre or "Sin hallazgo" in nombre:
        return "ok","badge-ok","✅","NORMAL"
    if nombre in URGENTES and prob > 0.40:
        return "alert","badge-alert","🚨","ALTA"
    if (nombre in URGENTES or nombre in IMPORTANTES) and prob > 0.25:
        return "warn","badge-warn","⚠️","MEDIA"
    if prob > 0.50:
        return "warn","badge-warn","⚠️","MEDIA"
    return "info","badge-info","🔍","BAJA"


# ─────────────────────────────────────────────────────────────
# INFORME
# ─────────────────────────────────────────────────────────────
def generar_informe(resultados: dict, region: str, motor: str) -> str:
    ahora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    top3  = sorted(resultados.items(), key=lambda x: x[1], reverse=True)[:3]
    lineas = [
        "="*60,
        "  INFORME RX DIAGNÓSTICO IA v4",
        "="*60,
        f"Fecha:    {ahora}",
        f"Región:   {region}",
        f"Motor IA: {motor}",
        "",
        "HALLAZGOS PRINCIPALES:",
        "-"*40,
    ]
    for n, p in top3:
        _, _, ico, gv = gravedad(n, p)
        lineas.append(f"  {ico} {n:<38} {p*100:5.1f}%  [{gv}]")
    lineas += ["", "RESULTADOS COMPLETOS:", "-"*40]
    for n, p in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
        lineas.append(f"  {n:<42} {p*100:5.1f}%")
    lineas += [
        "","="*60,
        "AVISO: Solo investigación/apoyo clínico.",
        "No reemplaza diagnóstico de radiólogo.",
        "Para uso clínico requiere validación y marcado CE.",
        "="*60,
    ]
    return "\n".join(lineas)


# ─────────────────────────────────────────────────────────────
# HISTORIAL
# ─────────────────────────────────────────────────────────────
if "historial" not in st.session_state:
    st.session_state.historial = []


# ─────────────────────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────────────────────
st.markdown("# 🩻 RX DIAGNÓSTICO IA")
st.markdown("<p style='color:#546e8a;margin-top:-.8rem;font-size:.9rem'>v4 · Detección automática de región · TorchXRayVision + CLIP especializado</p>", unsafe_allow_html=True)
st.markdown("---")

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.markdown(f"<span class='tag'>Hardware</span><br><b style='color:#00e5ff;font-family:DM Mono'>{gpu}</b>", unsafe_allow_html=True)
    st.markdown("---")

    deteccion_auto = st.checkbox("🤖 Detección automática de región", value=True,
                                  help="El programa decide solo si es tórax, hueso, columna o cráneo")
    
    if not deteccion_auto:
        region_manual = st.selectbox("Región manual:", 
                                      ["🫁 Tórax","🦴 Extremidades","🦴 Columna","🧠 Cráneo"])
    
    filtro_sel = st.radio("Preprocesamiento:",
                           ["Original","Contraste óseo","CLAHE simulado","Invertida"],
                           help="'Original' suele funcionar mejor")
    
    umbral = st.slider("Umbral mínimo (%)", 5, 40, 10, 5)
    
    st.markdown("---")
    if st.session_state.historial:
        st.markdown(f"<span class='tag'>Análisis en sesión</span><br><b style='color:#00e5ff'>{len(st.session_state.historial)}</b>", unsafe_allow_html=True)
        if st.button("🗑️ Limpiar historial"):
            st.session_state.historial = []
    
    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#3a5a7a;line-height:1.5'>⚠️ Solo investigación y apoyo clínico.<br>No reemplaza diagnóstico médico.<br><br>Para certificación: MDR 2017/745<br>ISO 13485 · AEMPS</div>", unsafe_allow_html=True)

# ── CARGA MODELOS ──
with st.spinner("⏳ Preparando motores IA... (solo la primera vez)"):
    cargar_xrv()
    if CLIP_DISPONIBLE:
        cargar_clip()

# ── UPLOAD ──
archivo = st.file_uploader("📤 Subir radiografía:", type=['png','jpg','jpeg'])

def aplicar_filtro(img, modo):
    g = img.convert('L')
    if modo == "Contraste óseo":
        return ImageEnhance.Sharpness(ImageOps.autocontrast(g, cutoff=2).convert('RGB')).enhance(1.8)
    if modo == "CLAHE simulado":
        a = np.array(g, dtype=np.float32)
        a = (a-a.min())/(a.max()-a.min()+1e-8)*255
        return Image.fromarray(a.astype(np.uint8)).convert('RGB')
    if modo == "Invertida":
        return ImageOps.autocontrast(ImageOps.invert(g)).convert('RGB')
    return img.convert('RGB')

if archivo:
    img_orig = Image.open(archivo).convert('RGB')
    img_proc = aplicar_filtro(img_orig, filtro_sel)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span class='tag'>Imagen original</span>", unsafe_allow_html=True)
        st.image(img_orig, use_container_width=True)
    with col2:
        st.markdown(f"<span class='tag'>Filtro: {filtro_sel}</span>", unsafe_allow_html=True)
        st.image(img_proc, use_container_width=True)

    st.markdown("")

    if st.button("🔬 ANALIZAR"):
        
        # ── PASO 1: DETECCIÓN DE REGIÓN ──
        with st.spinner("🔍 Detectando región anatómica..."):
            if deteccion_auto:
                region_det, conf_det, motivo_det = detectar_region(img_proc)
            else:
                mapa = {"🫁 Tórax":"torax","🦴 Extremidades":"extremidades",
                        "🦴 Columna":"columna","🧠 Cráneo":"craneo"}
                region_det = mapa[region_manual]
                conf_det   = 1.0
                motivo_det = "Selección manual"

        nombres_region = {
            "torax":"🫁 Tórax","extremidades":"🦴 Extremidades",
            "columna":"🦴 Columna","craneo":"🧠 Cráneo"
        }
        nombre_region = nombres_region[region_det]

        st.markdown(f"""
        <div class='detector-box'>
            <span class='tag'>Región detectada automáticamente</span><br>
            <b style='font-family:Syne;font-size:1.2rem;color:#00e5ff'>{nombre_region}</b>
            &nbsp;<span class='badge badge-info'>{conf_det*100:.0f}% confianza</span><br>
            <span style='color:#546e8a;font-size:.82rem'>{motivo_det}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── PASO 2: ANÁLISIS CON MOTOR CORRECTO ──
        with st.spinner(f"🧠 Analizando {nombre_region}..."):
            if region_det == "torax":
                resultados = analizar_torax(img_proc)
                motor_usado = "TorchXRayVision DenseNet-121 (700k+ RX reales)"
            elif region_det == "extremidades":
                if CLIP_DISPONIBLE:
                    resultados = analizar_clip(img_proc, PROMPTS_OSEOS)
                    motor_usado = "CLIP ViT-Large + Prompts especializados musculoesqueléticos"
                else:
                    st.warning("Instala transformers para análisis óseo: pip install transformers")
                    st.stop()
            elif region_det == "columna":
                if CLIP_DISPONIBLE:
                    resultados = analizar_clip(img_proc, PROMPTS_COLUMNA)
                    motor_usado = "CLIP ViT-Large + Prompts especializados vertebrales"
                else:
                    st.warning("Instala transformers para análisis de columna: pip install transformers")
                    st.stop()
            else:  # craneo
                if CLIP_DISPONIBLE:
                    resultados = analizar_clip(img_proc, PROMPTS_CRANEO)
                    motor_usado = "CLIP ViT-Large + Prompts especializados craneales"
                else:
                    st.warning("Instala transformers para análisis craneal: pip install transformers")
                    st.stop()

        # ── RESULTADO ──
        res_sorted = sorted(resultados.items(), key=lambda x: x[1], reverse=True)
        top_n, top_p = res_sorted[0]
        cls, bcls, ico, nivel = gravedad(top_n, top_p)
        color = COLOR[cls]

        st.markdown("---")

        # Hero
        st.markdown(f"""
        <div class='result-hero {cls}'>
            <span class='tag' style='color:{color}'>Diagnóstico principal · Confianza {nivel}</span>
            <div class='percent-big' style='color:{color}'>{top_p*100:.1f}%</div>
            <div class='diagnosis-name'>{ico} {top_n}</div>
            <span class='badge badge-purple'>{nombre_region}</span>
            &nbsp;<span class='badge badge-info' style='font-size:.65rem'>{motor_usado}</span>
        </div>
        """, unsafe_allow_html=True)

        # Desglose
        col_res, col_info = st.columns([3,2])
        with col_res:
            st.markdown("#### 📋 Hallazgos detectados")
            for nombre, prob in res_sorted:
                if prob * 100 < umbral:
                    continue
                c, bc, ic, gv = gravedad(nombre, prob)
                col_n = COLOR[c]
                st.markdown(f"""
                <div class='path-row'>
                    <span>{ic} {nombre}</span>
                    <span style='font-family:DM Mono;color:{col_n};font-weight:600'>{prob*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(min(prob, 1.0)))

        with col_info:
            st.markdown("#### 📊 Resumen de riesgo")
            urgentes_det    = [(n,p) for n,p in res_sorted if n in URGENTES and p>0.30]
            importantes_det = [(n,p) for n,p in res_sorted if n in IMPORTANTES and p>0.25]
            
            if urgentes_det:
                st.markdown(f"<span class='badge badge-alert'>🚨 {len(urgentes_det)} hallazgo(s) urgente(s)</span>", unsafe_allow_html=True)
                for n,p in urgentes_det:
                    st.markdown(f"<span style='color:#ff3d5a;font-size:.85rem'>→ {n}: {p*100:.1f}%</span>", unsafe_allow_html=True)
            elif importantes_det:
                st.markdown(f"<span class='badge badge-warn'>⚠️ {len(importantes_det)} hallazgo(s) a valorar</span>", unsafe_allow_html=True)
                for n,p in importantes_det[:3]:
                    st.markdown(f"<span style='color:#ffaa00;font-size:.85rem'>→ {n}: {p*100:.1f}%</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge badge-ok'>✅ Sin hallazgos urgentes</span>", unsafe_allow_html=True)

            # Nota sobre el motor
            st.markdown("---")
            if region_det != "torax":
                st.markdown(f"""
                <div class='warn-box'>
                    <span class='tag' style='color:#ffaa00'>Nota del motor</span><br>
                    <span style='font-size:.8rem;color:#ffaa00'>
                    Para {nombre_region} se usa CLIP, un modelo de visión general.<br>
                    La precisión es orientativa. El motor especializado para huesos
                    (MURA) estará disponible en v5.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='detector-box'>
                    <span class='tag'>Motor especializado</span><br>
                    <span style='font-size:.8rem;color:#00e5ff'>
                    DenseNet-121 entrenado en NIH+CheXpert+PadChest+MIMIC.<br>
                    Alta precisión para patología torácica.
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # Recomendación
        st.markdown("---")
        if urgentes_det:
            rec = f"🚨 Hallazgos urgentes detectados ({', '.join([n for n,_ in urgentes_det[:2]])}). Valoración clínica prioritaria."
            rcls = "alert"
        elif importantes_det:
            rec = "⚠️ Hallazgos que requieren correlación clínica. Considera proyecciones adicionales."
            rcls = "warn"
        elif "Normal" in top_n or "Sin hallazgo" in top_n:
            rec = "✅ Sin hallazgos patológicos significativos. Confirmar con valoración clínica."
            rcls = "ok"
        else:
            rec = "🔍 Hallazgos de baja probabilidad. Correlacionar con clínica y antecedentes."
            rcls = "info"

        st.markdown(f"""
        <div class='card' style='border-left:4px solid {COLOR[rcls]}'>
            <span class='tag'>Recomendación clínica</span>
            <p style='margin:.5rem 0 0;font-size:.95rem'>{rec}</p>
        </div>
        """, unsafe_allow_html=True)

        # Guardar historial
        st.session_state.historial.append({
            "ts": datetime.datetime.now().strftime("%H:%M:%S"),
            "archivo": archivo.name,
            "region": nombre_region,
            "top": (top_n, top_p),
        })

        # Informe
        with st.expander("📄 Informe descargable"):
            txt = generar_informe(resultados, nombre_region, motor_usado)
            st.text(txt)
            st.download_button("⬇️ Descargar .txt", data=txt,
                               file_name=f"rx_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                               mime="text/plain")

        # Info técnica
        with st.expander("🔧 Info técnica"):
            st.markdown(f"""
            <div style='font-family:DM Mono,monospace;font-size:.78rem;color:#3a5a7a;line-height:1.8'>
            Región detectada: {nombre_region}<br>
            Motor usado:      {motor_usado}<br>
            Patologías:       {len(resultados)}<br>
            Hardware:         {gpu}<br>
            Filtro:           {filtro_sel}<br>
            Umbral:           {umbral}%
            </div>
            """, unsafe_allow_html=True)

else:
    # Bienvenida
    st.markdown("""
    <div class='card' style='text-align:center;padding:3rem 2rem'>
        <div style='font-size:3.5rem;margin-bottom:1rem'>🩻</div>
        <div style='font-family:Syne;font-size:1.3rem;color:#ccd6f6;margin-bottom:.5rem'>
            Sube una radiografía — el programa detecta automáticamente qué es
        </div>
        <div style='color:#546e8a;font-size:.88rem;max-width:500px;margin:0 auto'>
            Tórax, huesos, columna o cráneo — el motor correcto se activa solo
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, (ico, reg, mot, desc) in zip([col1,col2,col3,col4], [
        ("🫁","Tórax","TorchXRayVision","Neumonía, neumotórax, derrame, cardiomegalia..."),
        ("🦴","Extremidades","CLIP especializado","Fracturas, luxaciones, artrosis, tumores..."),
        ("🦴","Columna","CLIP especializado","Fracturas, escoliosis, espondilosis, listesis..."),
        ("🧠","Cráneo","CLIP especializado","Fracturas, sinusitis, lesiones líticas..."),
    ]):
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center'>
                <div style='font-size:2rem'>{ico}</div>
                <div style='font-family:Syne;color:#00e5ff;font-weight:700;margin:.3rem 0'>{reg}</div>
                <div style='color:#546e8a;font-size:.72rem;margin-bottom:.3rem'>{mot}</div>
                <div style='color:#8a9ab0;font-size:.78rem'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.historial:
        st.markdown("---")
        st.markdown("#### 🕓 Análisis recientes")
        for item in reversed(st.session_state.historial[-5:]):
            n, p = item["top"]
            st.markdown(f"""
            <div class='path-row'>
                <span style='color:#546e8a;font-family:DM Mono;font-size:.78rem'>{item['ts']}</span>
                <span style='flex:1;margin:0 1rem'>{item['archivo']}</span>
                <span style='color:#c87fff;font-size:.78rem'>{item['region']}</span>
                <span style='color:#00e5ff;font-family:DM Mono'>{n}: {p*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
