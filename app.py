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

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="DSP Workbench", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; } 
    .block-container { padding-top: 1rem; }
    /* Estilo tipo 'Rack' para el monitor */
    .dsp-monitor { 
        background-color: #222; color: #0f0; 
        padding: 8px 12px; border-radius: 4px; 
        font-family: 'Consolas', monospace; font-size: 0.9em;
        border: 1px solid #444; margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO ---
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
    st.session_state.fs = 0
    st.session_state.file_name = ""

# --- 3. CALLBACKS DE CARGA ---
def new_file_loaded(data, fs, name):
    st.session_state.audio_data = data
    st.session_state.fs = fs
    st.session_state.file_name = name

def load_uploaded():
    if st.session_state.uploader:
        d, fs = load_audio(st.session_state.uploader)
        new_file_loaded(d, fs, st.session_state.uploader.name)

def load_example():
    path = os.path.join("examples", st.session_state.ex_selector)
    if os.path.exists(path):
        d, fs = load_audio(path)
        new_file_loaded(d, fs, st.session_state.ex_selector)

# --- 4. UTILIDADES DSP & VISUAL ---
def normalize_visuals(data):
    """Normaliza señal a [-1, 1] solo para graficar"""
    mx = np.max(np.abs(data))
    if mx > 0: return data / mx
    return data

def safe_downsample(data, max_points=2000):
    """
    Reduce drásticamente los puntos para evitar que el navegador explote (Crash Fix).
    Si hay 3 millones de puntos, toma 1 de cada 1500.
    """
    if len(data) > max_points:
        step = int(np.ceil(len(data) / max_points))
        return data[::step]
    return data

def render_player(audio_bytes, fs, unique_id):
    """Reproductor simple y robusto"""
    b64 = base64.b64encode(audio_bytes.read()).decode()
    html_id = f"audio_{unique_id}"
    
    html = f"""
    <div class="dsp-monitor">OUTPUT: {fs} Hz | STATUS: READY</div>
    <audio id="{html_id}" controls autoplay style="width:100%;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.components.v1.html(html, height=85)

# --- 5. INTERFAZ ---
st.title("🎛️ Ingeniería de Señales: DSP Workbench")

# INPUT
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
    st.info("⚠️ Carga una señal para iniciar.")
    st.stop()

# DATOS
raw_data = st.session_state.audio_data
fs_in = st.session_state.fs

# PROCESAMIENTO
st.sidebar.markdown("---")
use_loop = st.sidebar.checkbox("Modo Loop (15s)", value=True, help="Evita saturar la memoria.")

if use_loop:
    mid = len(raw_data) // 2
    win = 15 * fs_in
    start = max(0, mid - (win//2))
    end = min(len(raw_data), start + win)
    work_data = raw_data[start:end]
else:
    work_data = raw_data

# PARÁMETROS
c1, c2 = st.sidebar.columns(2)
L = c1.number_input("Upsample (L)", 1, 8, 1)
M = c2.number_input("Downsample (M)", 1, 8, 1)

st.sidebar.subheader("Banco de Filtros")
bands = ["Sub", "Bass", "LoMid", "HiMid", "Pres", "Brill"]
freq_ranges = ["16-60", "60-250", "250-2k", "2k-4k", "4k-6k", "6k-16k"]
keys = ["Sub-Bass", "Bass", "Low Mids", "High Mids", "Presence", "Brilliance"]

gains = {}
cols = st.sidebar.columns(3)
for i, (label, k) in enumerate(zip(bands, keys)):
    with cols[i%3]:
        gains[k] = st.slider(label, -15, 15, 0, help=f"{freq_ranges[i]} Hz")

# --- DSP ENGINE ---
# 1. Resampling
resampled, fs_out = change_sampling_rate(work_data, fs_in, M, L)
# 2. Equalization
processed = apply_equalizer(resampled, fs_out, gains)

# --- VISUALIZACIÓN ---
st.divider()

# OPCIONES DE VISUALIZACIÓN
viz_col1, viz_col2 = st.columns([3, 1])
with viz_col2:
    st.markdown("##### Configuración Gráfica")
    freq_unit = st.radio("Unidad Frecuencia:", ["Hz (Ciclos/s)", "rad/s (Angular)"])
    
    # Factor de conversión para el eje X
    x_factor = 2 * np.pi if "rad/s" in freq_unit else 1.0
    x_label = "Frecuencia Angular (rad/s)" if "rad/s" in freq_unit else "Frecuencia (Hz)"

with viz_col1:
    tab1, tab2 = st.tabs(["📈 Dominio Tiempo", "🌊 Dominio Frecuencia"])

    # Preparamos datos visuales (Normalizados y Downsampled)
    # CRASH FIX: Usamos safe_downsample para garantizar pocos puntos
    v_in = safe_downsample(normalize_visuals(work_data))
    v_out = safe_downsample(normalize_visuals(processed))
    
    # Ejes de tiempo
    t_in = np.linspace(0, len(v_in)/fs_in, len(v_in))
    t_out = np.linspace(0, len(v_out)/fs_out, len(v_out))

    # --- GRÁFICA TIEMPO ---
    with tab1:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=t_in, y=v_in, name="Original", 
                                 line=dict(color='#888', width=1), opacity=0.5))
        fig_t.add_trace(go.Scatter(x=t_out, y=v_out, name="Procesada", 
                                 line=dict(color='#0f0', width=1.5)))
        
        # LAYOUT INGENIERIL (Ejes visibles)
        fig_t.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
            height=320,
            title="Comparativa Temporal (Normalizada)",
            xaxis=dict(
                title="Tiempo (s)", 
                showline=True, mirror=True, linecolor='white', linewidth=2,
                showgrid=True, gridcolor='#444'
            ),
            yaxis=dict(
                title="Amplitud (Normalizada)", 
                showline=True, mirror=True, linecolor='white', linewidth=2,
                showgrid=True, gridcolor='#444'
            )
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # --- GRÁFICA FRECUENCIA ---
    with tab2:
        # Calcular FFT (Usamos solo un slice seguro si es muy grande para evitar crash en cálculo)
        limit_fft = min(len(work_data), 200000) # Max 200k muestras para FFT visual
        
        f_in_raw, m_in_raw = compute_fft(work_data[:limit_fft], fs_in)
        f_out_raw, m_out_raw = compute_fft(processed[:limit_fft], fs_out)
        
        # Downsample visual
        vf_in = safe_downsample(f_in_raw)
        vm_in = safe_downsample(20*np.log10(m_in_raw+1e-9))
        
        vf_out = safe_downsample(f_out_raw)
        vm_out = safe_downsample(20*np.log10(m_out_raw+1e-9))

        # Conversión a Angular si aplica
        vf_in = vf_in * x_factor
        vf_out = vf_out * x_factor

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=vf_in, y=vm_in, name="Original", 
                                 line=dict(color='#888', width=1)))
        fig_f.add_trace(go.Scatter(x=vf_out, y=vm_out, name="Procesada", 
                                 fill='tozeroy', line=dict(color='#00ffff', width=1)))
        
        # LAYOUT INGENIERIL
        fig_f.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
            height=320,
            title="Espectro de Magnitud",
            xaxis=dict(
                title=x_label, type="log",
                showline=True, mirror=True, linecolor='white', linewidth=2,
                showgrid=True, gridcolor='#444'
            ),
            yaxis=dict(
                title="Magnitud (dB)",
                showline=True, mirror=True, linecolor='white', linewidth=2,
                showgrid=True, gridcolor='#444'
            )
        )
        st.plotly_chart(fig_f, use_container_width=True)

# --- OUTPUT AUDIO ---
with viz_col2:
    st.markdown("### 💾 Salida")
    
    # Preparar audio final
    audio_out = np.nan_to_num(processed)
    pk = np.max(np.abs(audio_out))
    if pk > 0: audio_out /= pk
    audio_out = np.clip(audio_out, -1.0, 1.0)
    
    buffer = io.BytesIO()
    write(buffer, fs_out, (audio_out * 32767).astype(np.int16))
    
    # ID único aleatorio para forzar recarga del reproductor
    render_player(buffer, fs_out, str(uuid.uuid4()))
    
    st.download_button("Descargar WAV", buffer, f"dsp_out_{fs_out}Hz.wav", "audio/wav")

# Limpieza memoria
del resampled, processed, audio_out, buffer
gc.collect()
