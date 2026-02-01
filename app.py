import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import io
import os
import gc
import base64
import uuid
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# CONFIGURACIÓN
st.set_page_config(page_title="DSP Workbench", layout="wide", page_icon="🎛️")
st.markdown("""<style>.block-container { padding-top: 1rem; }</style>""", unsafe_allow_html=True)

# ESTADO
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
    st.session_state.fs = 44100
    st.session_state.file_id = str(uuid.uuid4())

# CARGA DE ARCHIVOS
def load_new_file(data, fs):
    st.session_state.audio_data = data
    st.session_state.fs = fs
    st.session_state.file_id = str(uuid.uuid4())

# UI LATERAL
st.sidebar.title("🎛️ DSP Config")
mode = st.sidebar.radio("Fuente", ["Ejemplo", "Subir"], horizontal=True)

if mode == "Subir":
    upl = st.sidebar.file_uploader("WAV", type=["wav"])
    if upl: 
        d, fs = load_audio(upl)
        load_new_file(d, fs)
else:
    if os.path.exists("examples"):
        exs = [f for f in os.listdir("examples") if f.endswith('.wav')]
        sel = st.sidebar.selectbox("Archivo", exs)
        if sel:
            d, fs = load_audio(os.path.join("examples", sel))
            load_new_file(d, fs)

if st.session_state.audio_data is None:
    st.warning("Carga un archivo para comenzar.")
    st.stop()

# PROCESAMIENTO
raw = st.session_state.audio_data
fs_in = st.session_state.fs

# Loop 15s
use_loop = st.sidebar.checkbox("Modo Loop (15s)", True)
if use_loop and len(raw) > 15*fs_in:
    start = len(raw)//2 - (7*fs_in)
    raw = raw[start : start + 15*fs_in]

# Controles
c1, c2 = st.sidebar.columns(2)
L = c1.number_input("Upsample L", 1, 8, 1)
M = c2.number_input("Downsample M", 1, 8, 1)

gains = {}
st.sidebar.subheader("Ecualizador")
cols = st.sidebar.columns(3)
keys = ["Sub-Bass", "Bass", "Low Mids", "High Mids", "Presence", "Brilliance"]
for i, k in enumerate(keys):
    gains[k] = cols[i%3].slider(k[:3], -15, 15, 0)

# --- MOTOR DSP ---
resampled, fs_out = change_sampling_rate(raw, fs_in, M, L)
processed = apply_equalizer(resampled, fs_out, gains)

# --- VISUALIZACIÓN ---
st.divider()
t1, t2 = st.tabs(["Tiempo", "Frecuencia"])

def safe_downsample(arr, max_pts=3000):
    if len(arr) > max_pts: return arr[::int(len(arr)/max_pts)]
    return arr

with t1:
    fig_t = go.Figure()
    # Tiempo: Normalizamos solo para visualización
    v_in = safe_downsample(raw)
    v_out = safe_downsample(processed)
    
    fig_t.add_trace(go.Scatter(y=v_in, name="In", line=dict(color='gray', width=1), opacity=0.5))
    fig_t.add_trace(go.Scatter(y=v_out, name="Out", line=dict(color='#00FF00', width=1)))
    fig_t.update_layout(height=300, margin=dict(t=20,b=20), template="plotly_dark", title="Dominio del Tiempo")
    st.plotly_chart(fig_t, use_container_width=True)

with t2:
    # FFT
    limit = 100000
    fi, mi = compute_fft(raw[:limit], fs_in)
    fo, mo = compute_fft(processed[:limit], fs_out)
    
    # --- SANITIZACIÓN CRÍTICA PARA GRÁFICOS ---
    # 1. Eliminar 0 Hz (DC) y Frecuencias Negativas
    mask_i = fi > 1.0 # Solo frecuencias > 1 Hz
    mask_o = fo > 1.0
    
    fi, mi = fi[mask_i], mi[mask_i]
    fo, mo = fo[mask_o], mo[mask_o]
    
    # 2. Conversión a dB Segura (Clip en -100 dB)
    # Evita log(0) y valores de -300 dB que rompen la escala
    mi_db = 20 * np.log10(np.maximum(mi, 1e-5)) 
    mo_db = 20 * np.log10(np.maximum(mo, 1e-5))
    
    # 3. Downsample visual
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=safe_downsample(fi), y=safe_downsample(mi_db), name="Original", line=dict(color='gray')))
    fig_f.add_trace(go.Scatter(x=safe_downsample(fo), y=safe_downsample(mo_db), name="Procesada", line=dict(color='cyan')))
    
    # Líneas de referencia
    for b in [60, 250, 2000, 4000, 6000]:
        fig_f.add_vline(x=b, line_dash="dash", line_color="yellow", opacity=0.5)

    fig_f.update_layout(
        height=350, template="plotly_dark", title="Espectro (dB)",
        xaxis=dict(type="log", title="Hz", range=[np.log10(20), np.log10(fs_in/2)]), # Zoom útil: 20Hz a Nyquist
        yaxis=dict(title="dB", range=[-80, 20]) # Rango fijo útil para audio
    )
    st.plotly_chart(fig_f, use_container_width=True)

# --- AUDIO OUT ---
# Normalizar para reproducción
final_audio = processed / (np.max(np.abs(processed)) + 1e-6)
final_audio = np.clip(final_audio, -1.0, 1.0)
buffer = io.BytesIO()
write(buffer, fs_out, (final_audio * 32767).astype(np.int16))

st.audio(buffer, format="audio/wav")
