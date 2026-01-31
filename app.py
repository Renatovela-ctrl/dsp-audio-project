import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
import os
import gc
import base64
import uuid
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- CONFIGURACIÓN ESTRUCTURAL ---
st.set_page_config(page_title="DSP Live Workbench", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; } 
    .block-container { padding-top: 1rem; }
    .dsp-monitor { 
        background-color: #1e1e1e; color: #00ff00; 
        padding: 10px; border-radius: 5px; font-family: monospace; 
        margin-bottom: 10px; border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- GESTIÓN DE ESTADO (SESSION STATE) ---
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
    st.session_state.fs = 0
    st.session_state.file_id = str(uuid.uuid4()) # ID único del archivo actual
    st.session_state.file_name = ""

# --- LOGICA DE CARGA (CALLBACKS) ---
def new_file_loaded(data, fs, name):
    st.session_state.audio_data = data
    st.session_state.fs = fs
    st.session_state.file_name = name
    # CAMBIAMOS EL ID DEL ARCHIVO: Esto le dice a Plotly "Hey, es un archivo nuevo, resetea el zoom"
    st.session_state.file_id = str(uuid.uuid4()) 

def load_uploaded():
    if st.session_state.uploader:
        d, fs = load_audio(st.session_state.uploader)
        new_file_loaded(d, fs, st.session_state.uploader.name)

def load_example():
    path = os.path.join("examples", st.session_state.ex_selector)
    if os.path.exists(path):
        d, fs = load_audio(path)
        new_file_loaded(d, fs, st.session_state.ex_selector)

# --- UTILIDADES DSP & VISUAL ---
def normalize_visuals(data):
    """Normaliza para visualización: Rango [-1, 1]"""
    # Evita que una señal suene fuerte y la otra se vea pequeña
    mx = np.max(np.abs(data))
    if mx > 0:
        return data / mx
    return data

def downsample(data, max_points=3000):
    if len(data) > max_points:
        return data[::len(data)//max_points]
    return data

def render_player(audio_bytes, fs, unique_token):
    """Reproductor que respeta la sesión"""
    b64 = base64.b64encode(audio_bytes.read()).decode()
    # Usamos el unique_token (file_id) para guardar el tiempo
    # Si cambias de archivo (token nuevo), el tiempo empieza de 0.
    # Si ecualizas (token igual), el tiempo se mantiene.
    storage_key = f"time_{unique_token}"
    html_id = f"audio_{unique_token}" # ID único para el DOM
    
    html = f"""
    <div class="dsp-monitor">Output Monitor: {fs} Hz | Active</div>
    <audio id="{html_id}" controls autoplay style="width:100%;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    <script>
        (function() {{
            var a = document.getElementById('{html_id}');
            var k = '{storage_key}';
            a.onloadedmetadata = function() {{
                var s = sessionStorage.getItem(k);
                if(s && s!=="null") {{
                    var t = parseFloat(s);
                    if(t < a.duration) a.currentTime = t;
                }}
                a.play().catch(e=>console.log("Autoplay waiting"));
            }};
            a.ontimeupdate = function() {{ sessionStorage.setItem(k, a.currentTime); }};
        }})();
    </script>
    """
    st.components.v1.html(html, height=100)

# --- INTERFAZ ---
st.title("🎛️ Ingeniería de Señales: DSP Workbench")

# 1. INPUT
col_in1, col_in2 = st.sidebar.columns(2)
mode = col_in1.radio("Fuente", ["Ejemplo", "Subir"], label_visibility="collapsed")

if mode == "Subir":
    st.sidebar.file_uploader("WAV File", type=["wav"], key="uploader", on_change=load_uploaded)
else:
    if os.path.exists("examples"):
        files = [f for f in os.listdir("examples") if f.endswith('.wav')]
        if files:
            st.sidebar.selectbox("Seleccionar", files, key="ex_selector", on_change=load_example)

if st.session_state.audio_data is None:
    st.info("⚠️ Carga una señal para iniciar el análisis.")
    st.stop()

# Recuperar datos
raw_data = st.session_state.audio_data
fs_in = st.session_state.fs

# 2. PROCESAMIENTO
st.sidebar.markdown("### ⚙️ Procesamiento Multitasa")
use_loop = st.sidebar.checkbox("Modo Análisis (Loop 15s)", value=True, help="Analiza un segmento corto para respuesta inmediata.")

# Selección de datos
if use_loop:
    mid = len(raw_data) // 2
    win = 15 * fs_in
    start = max(0, mid - (win//2))
    end = min(len(raw_data), start + win)
    work_data = raw_data[start:end]
    st.sidebar.caption(f"Analizando tramo: {start/fs_in:.1f}s - {end/fs_in:.1f}s")
else:
    work_data = raw_data
    st.sidebar.warning("Procesando archivo completo (Lento)")

# Controles M/L
c1, c2 = st.sidebar.columns(2)
L = c1.number_input("Upsample (L)", 1, 8, 1)
M = c2.number_input("Downsample (M)", 1, 8, 1)

# Ecualizador
st.sidebar.markdown("### 🎚️ Banco de Filtros")
gains = {}
bands = ["Sub (16-60)", "Bass (60-250)", "LoMid (250-2k)", "HiMid (2k-4k)", "Pres (4k-6k)", "Brill (6k-16k)"]
keys = ["Sub-Bass", "Bass", "Low Mids", "High Mids", "Presence", "Brilliance"]

cols = st.sidebar.columns(3)
for i, (b_label, b_key) in enumerate(zip(bands, keys)):
    with cols[i%3]:
        gains[b_key] = st.slider(b_label.split()[0], -15, 15, 0, key=f"eq_{i}", help=b_label)

# --- DSP ENGINE ---
# 1. Resampling
resampled, fs_out = change_sampling_rate(work_data, fs_in, M, L)
# 2. Equalization
processed = apply_equalizer(resampled, fs_out, gains)

# --- VISUALIZACIÓN ---
st.divider()
c_viz, c_out = st.columns([3, 1])

with c_viz:
    tab1, tab2 = st.tabs(["📈 Análisis Temporal", "🌊 Análisis Espectral"])
    
    # Pre-cálculo de visuales
    # IMPORTANTE: Normalizamos AMBOS para que se vean comparables en amplitud
    v_in = downsample(normalize_visuals(work_data))
    v_out = downsample(normalize_visuals(processed))
    
    t_in = np.linspace(0, len(v_in)/fs_in, len(v_in))
    t_out = np.linspace(0, len(v_out)/fs_out, len(v_out))

    # --- GRAFICA TEMPORAL ---
    with tab1:
        fig_t = go.Figure()
        # Original en gris tenue
        fig_t.add_trace(go.Scatter(x=t_in, y=v_in, name="Original (Norm)", 
                                 line=dict(color='gray', width=1), opacity=0.5))
        # Procesada en verde brillante
        fig_t.add_trace(go.Scatter(x=t_out, y=v_out, name="Procesada (Norm)", 
                                 line=dict(color='#00ff00', width=1.5)))
        
        # LA CLAVE DEL ZOOM: uirevision
        # Usamos st.session_state.file_id. 
        # Si mueves el slider, file_id NO cambia -> ZOOM SE QUEDA.
        # Si cambias archivo, file_id cambia -> ZOOM RESET.
        fig_t.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            uirevision=st.session_state.file_id, 
            title="Comparativa en el Tiempo (Normalizada)"
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # --- GRAFICA ESPECTRAL ---
    with tab2:
        f_in, m_in = compute_fft(work_data, fs_in)
        f_out, m_out = compute_fft(processed, fs_out)
        
        vf_in = downsample(f_in)
        vm_in = downsample(20*np.log10(m_in+1e-9))
        vf_out = downsample(f_out)
        vm_out = downsample(20*np.log10(m_out+1e-9))

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=vf_in, y=vm_in, name="Espectro Original", 
                                 line=dict(color='gray'), opacity=0.5))
        fig_f.add_trace(go.Scatter(x=vf_out, y=vm_out, name="Espectro Procesado", 
                                 fill='tozeroy', line=dict(color='#00ff00')))
        
        fig_f.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            xaxis_type="log",
            uirevision=st.session_state.file_id, # El mismo truco para el zoom
            title="Espectro de Magnitud (Banda Base)"
        )
        st.plotly_chart(fig_f, use_container_width=True)

with c_out:
    # --- SALIDA DE AUDIO ---
    # Normalización final para audio (Ojo: distinta a la visual)
    audio_out = np.nan_to_num(processed)
    peak = np.max(np.abs(audio_out))
    if peak > 0: audio_out /= peak # Normalizar a 0dBFS
    audio_out = np.clip(audio_out, -1.0, 1.0)
    
    # Buffer
    buffer = io.BytesIO()
    write(buffer, fs_out, (audio_out * 32767).astype(np.int16))
    
    # Reproductor
    render_player(buffer, fs_out, st.session_state.file_id)
    
    st.divider()
    st.download_button("💾 Descargar .WAV", buffer, f"processed_{st.session_state.file_id}.wav", "audio/wav")

# Garbage Collection explícito
del resampled, processed, audio_out, buffer
gc.collect()
