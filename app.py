import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
import os
import gc
import base64
import time
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="DSP Live Studio", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; }
    .block-container { padding-top: 1rem; }
    .dsp-card { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO AVANZADA ---
# Inicializamos variables para guardar las FIGURAS completas
if 'fig_time' not in st.session_state:
    st.session_state.fig_time = None
if 'fig_freq' not in st.session_state:
    st.session_state.fig_freq = None
# Variables de datos de audio
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'audio_fs' not in st.session_state:
    st.session_state.audio_fs = 0
if 'current_filename' not in st.session_state:
    st.session_state.current_filename = ""

# --- 3. CALLBACKS DE CARGA (Para evitar recargas accidentales) ---
def load_file_callback():
    if st.session_state.uploader is not None:
        data, fs = load_audio(st.session_state.uploader)
        st.session_state.audio_data = data
        st.session_state.audio_fs = fs
        st.session_state.current_filename = st.session_state.uploader.name
        # Resetear figuras al cambiar de archivo
        st.session_state.fig_time = None 
        st.session_state.fig_freq = None
        gc.collect()

def load_example_callback():
    fname = st.session_state.example_selector
    path = os.path.join("examples", fname)
    if os.path.exists(path):
        data, fs = load_audio(path)
        st.session_state.audio_data = data
        st.session_state.audio_fs = fs
        st.session_state.current_filename = fname
        # Resetear figuras
        st.session_state.fig_time = None
        st.session_state.fig_freq = None
        gc.collect()

# --- 4. UTILITARIOS ---
def downsample_visuals(data, max_points=2000):
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

def render_smart_player(audio_bytes, fs, unique_id):
    """Reproductor persistente v3.0"""
    b64 = base64.b64encode(audio_bytes.read()).decode()
    html_id = "persistent_player"
    storage_key = f"t_{unique_id}" # Clave corta y única por canción
    
    html = f"""
    <div class="dsp-card">
        <div style="font-size:0.8em; color:#666; margin-bottom:5px;">Monitor ({fs} Hz)</div>
        <audio id="{html_id}" controls autoplay style="width: 100%; height: 35px;">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    </div>
    <script>
        (function() {{
            var a = document.getElementById('{html_id}');
            var k = '{storage_key}';
            a.onloadedmetadata = function() {{
                var s = sessionStorage.getItem(k);
                if (s && s!=="null") {{
                    var t = parseFloat(s);
                    if (t < a.duration && t > 0) a.currentTime = t;
                }}
                a.play().catch(e=>console.log("Autoplay waiting"));
            }};
            a.ontimeupdate = function() {{ if(a.currentTime>0) sessionStorage.setItem(k, a.currentTime); }};
        }})();
    </script>
    """
    st.components.v1.html(html, height=100)

# --- 5. INTERFAZ ---
st.title("🎛️ DSP Live Studio")

# BARRA LATERAL
st.sidebar.header("1. Fuente")
input_type = st.sidebar.radio("Tipo:", ["Ejemplo", "Subir"], horizontal=True)

if input_type == "Subir":
    st.sidebar.file_uploader("WAV", type=["wav"], key="uploader", on_change=load_file_callback)
else:
    if os.path.exists("examples"):
        files = [f for f in os.listdir("examples") if f.endswith('.wav')]
        if files:
            st.sidebar.selectbox("Pista:", files, key="example_selector", on_change=load_example_callback)

# --- VALIDACIÓN DE DATOS ---
if st.session_state.audio_data is None:
    st.info("👈 Carga un archivo para empezar.")
    st.stop()

# Recuperar datos del estado
full_data = st.session_state.audio_data
original_fs = st.session_state.audio_fs
fname = st.session_state.current_filename

st.sidebar.caption(f"Track: **{fname}** | {original_fs} Hz")

# --- 6. CONTROLES ---
st.sidebar.markdown("---")
use_loop = st.sidebar.toggle("⚡ Modo Loop (15s)", value=True)

if use_loop:
    mid = len(full_data) // 2
    window = 15 * original_fs
    start = max(0, mid - (window // 2))
    end = min(len(full_data), start + window)
    working_data = full_data[start:end]
else:
    working_data = full_data

st.sidebar.subheader("DSP Settings")
c1, c2 = st.sidebar.columns(2)
L = c1.number_input("Upsample (L)", 1, 8, 1)
M = c2.number_input("Downsample (M)", 1, 8, 1)

st.sidebar.subheader("Ecualizador")
cols = st.sidebar.columns(3)
bands = ["Sub", "Bass", "LoMid", "HiMid", "Pres", "Brill"]
gains = {}
for i, b in enumerate(bands):
    with cols[i%3]:
        gains[b] = st.slider(b, -15, 15, 0, key=f"sl_{i}")

dsp_gains = {
    "Sub-Bass": gains["Sub"], "Bass": gains["Bass"], "Low Mids": gains["LoMid"],
    "High Mids": gains["HiMid"], "Presence": gains["Pres"], "Brilliance": gains["Brill"]
}

# --- 7. PROCESAMIENTO ---
resampled, new_fs = change_sampling_rate(working_data, original_fs, M, L)
processed = apply_equalizer(resampled, new_fs, dsp_gains)

# --- 8. VISUALIZACIÓN "STATEFUL" (LA CLAVE DEL ZOOM) ---
col_viz, col_play = st.columns([3, 2])

with col_viz:
    t1, t2 = st.tabs(["Tiempo", "Frecuencia"])
    
    # Preparar datos visuales
    vp = downsample_visuals(processed, 1500)
    tx = np.linspace(0, len(vp)/new_fs, len(vp))
    
    f, m = compute_fft(processed, new_fs)
    vf = downsample_visuals(f, 1500)
    vm = downsample_visuals(20*np.log10(m+1e-9), 1500)

    # --- PESTAÑA TIEMPO ---
    with t1:
        # Si la figura no existe (primera vez o cambio de archivo), la creamos
        if st.session_state.fig_time is None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=vp, x=tx, line=dict(color='#00cc96', width=1.5), name="Señal"))
            fig.update_layout(
                height=250, margin=dict(t=10,b=10,l=10,r=10), showlegend=False,
                uirevision="constant" # <--- ESTO MANTIENE EL ZOOM
            )
            st.session_state.fig_time = fig
        else:
            # Si YA existe, solo actualizamos los datos X e Y
            # Esto NO toca el 'layout' (donde vive el zoom)
            st.session_state.fig_time.data[0].y = vp
            st.session_state.fig_time.data[0].x = tx
        
        # Renderizamos la figura guardada en sesión
        st.plotly_chart(st.session_state.fig_time, use_container_width=True, key="p_time")

    # --- PESTAÑA FRECUENCIA ---
    with t2:
        if st.session_state.fig_freq is None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vf, y=vm, fill='tozeroy', name="FFT"))
            fig.update_layout(
                height=250, margin=dict(t=10,b=10,l=10,r=10), showlegend=False,
                xaxis_type="log",
                uirevision="constant" # <--- ESTO MANTIENE EL ZOOM
            )
            st.session_state.fig_freq = fig
        else:
            # Actualizamos solo datos
            st.session_state.fig_freq.data[0].x = vf
            st.session_state.fig_freq.data[0].y = vm
            
        st.plotly_chart(st.session_state.fig_freq, use_container_width=True, key="p_freq")

# --- 9. AUDIO ---
with col_play:
    # Botón para resetear zoom manualmente (Opcional, pero útil)
    if st.button("🔄 Resetear Zoom"):
        st.session_state.fig_time = None
        st.session_state.fig_freq = None
        st.rerun()

    # Procesado final
    clean = np.nan_to_num(processed)
    mx = np.max(np.abs(clean))
    if mx > 0: clean /= mx
    clean = np.clip(clean, -1.0, 1.0)
    
    wav_io = io.BytesIO()
    write(wav_io, new_fs, (clean * 32760).astype(np.int16))
    wav_io.seek(0)
    
    render_smart_player(wav_io, new_fs, fname)
    
    st.markdown("---")
    st.download_button("⬇️ Descargar", wav_io, f"dsp_{fname}", "audio/wav")

del resampled, processed, clean, wav_io
gc.collect()
