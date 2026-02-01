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
# Importamos las funciones manuales del core
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="DSP Workbench - UCuenca", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; } 
    .block-container { padding-top: 1rem; }
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
    st.session_state.file_id = str(uuid.uuid4())

# --- 3. CALLBACKS ---
def new_file_loaded(data, fs, name):
    st.session_state.audio_data = data
    st.session_state.fs = fs
    st.session_state.file_name = name
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

# --- 4. UTILIDADES VISUALES ---
def normalize_visuals(data):
    mx = np.max(np.abs(data))
    if mx > 0: return data / mx
    return data

def safe_downsample(data, max_points=2000):
    if len(data) > max_points:
        step = int(np.ceil(len(data) / max_points))
        return data[::step]
    return data

def render_player(audio_bytes, fs, unique_id):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    html_id = f"audio_{unique_id}"
    storage_key = f"time_{unique_id}"
    
    html = f"""
    <div class="dsp-monitor">OUTPUT: {fs} Hz | STATUS: READY</div>
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
    st.components.v1.html(html, height=85)

# --- 5. INTERFAZ PRINCIPAL ---
st.title("🎛️ Conversor de Frecuencia y Ecualizador - Sistemas Lineales")
st.caption("Proyecto Integrador: Israel Méndez, Daniel Molina, Renato Vela")

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

# PROCESAMIENTO LOOP
st.sidebar.markdown("---")
use_loop = st.sidebar.checkbox("Modo Loop (15s)", value=True, help="Optimiza rendimiento.")

if use_loop:
    mid = len(raw_data) // 2
    win = 15 * fs_in
    start = max(0, mid - (win//2))
    end = min(len(raw_data), start + win)
    work_data = raw_data[start:end]
else:
    work_data = raw_data

# CONTROLES
c1, c2 = st.sidebar.columns(2)
L = c1.number_input("Upsample (L)", 1, 8, 1)
M = c2.number_input("Downsample (M)", 1, 8, 1)

st.sidebar.subheader("Ecualizador (6 Bandas)")
bands = ["Sub", "Bass", "LoMid", "HiMid", "Pres", "Brill"]
keys = ["Sub-Bass", "Bass", "Low Mids", "High Mids", "Presence", "Brilliance"]
gains = {}
cols = st.sidebar.columns(3)
for i, (label, k) in enumerate(zip(bands, keys)):
    with cols[i%3]:
        gains[k] = st.slider(label, -15, 15, 0, key=f"eq_{i}")

# --- MOTOR DSP ---
# 1. Cambio de Tasa (Manual)
resampled, fs_out = change_sampling_rate(work_data, fs_in, M, L)
# 2. Ecualización (Manual)
processed = apply_equalizer(resampled, fs_out, gains)

# --- VISUALIZACIÓN ---
st.divider()
viz_mode = st.radio("Modo Visual:", ["🛠️ Análisis Completo", "📖 Teórico (Stem Plot)"], horizontal=True)

if viz_mode == "🛠️ Análisis Completo":
    col_opt, _ = st.columns([1, 4])
    with col_opt:
        f_unit = st.radio("Eje X:", ["Hz", "rad/s"])
        x_mult = 2*np.pi if f_unit == "rad/s" else 1.0

    t1, t2 = st.tabs(["Tiempo", "Frecuencia"])
    
    # Datos visuales
    v_in = safe_downsample(normalize_visuals(work_data))
    v_out = safe_downsample(normalize_visuals(processed))
    t_axis_in = np.linspace(0, len(v_in)/fs_in, len(v_in))
    t_axis_out = np.linspace(0, len(v_out)/fs_out, len(v_out))

    with t1:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=t_axis_in, y=v_in, name="In", line=dict(color='gray', width=1), opacity=0.5))
        fig_t.add_trace(go.Scatter(x=t_axis_out, y=v_out, name="Out", line=dict(color='#0f0', width=1.5)))
        fig_t.update_layout(
            template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10),
            title="Comparativa Temporal", uirevision=st.session_state.file_id
        )
        st.plotly_chart(fig_t, use_container_width=True)

with t2:
        # Calcular FFT
        limit = min(len(work_data), 100000)
        fi, mi = compute_fft(work_data[:limit], fs_in)
        fo, mo = compute_fft(processed[:limit], fs_out)
        
        # Conversión a dB
        mi_db = 20*np.log10(mi + 1e-9)
        mo_db = 20*np.log10(mo + 1e-9)

        # Downsample visual para no saturar el navegador
        vi_f = safe_downsample(fi) * x_mult
        vi_m = safe_downsample(mi_db)
        vo_f = safe_downsample(fo) * x_mult
        vo_m = safe_downsample(mo_db)

        fig_f = go.Figure()
        
        # Trazos de señal
        fig_f.add_trace(go.Scatter(x=vi_f, y=vi_m, name="Original", line=dict(color='gray', width=1), opacity=0.7))
        fig_f.add_trace(go.Scatter(x=vo_f, y=vo_m, name="Procesada", fill='tozeroy', line=dict(color='cyan', width=1.5)))
        
        # --- LÍNEAS DE SEPARACIÓN DE BANDAS ---
        boundaries = [60, 250, 2000, 4000, 6000]
        
        for b in boundaries:
            pos = b * x_mult
            # Línea vertical amarilla
            fig_f.add_vline(
                x=pos, 
                line_width=1.5, 
                line_dash="dash", 
                line_color="#FFFF00",
                opacity=0.7
            )
            # Etiqueta de texto (solo si está en Hz para no amontonar)
            if f_unit == "Hz":
                fig_f.add_annotation(
                    x=pos, 
                    y=0, # Posición en Y (ajustable)
                    text=f"{b}", 
                    showarrow=False, 
                    yshift=10,
                    font=dict(color="#FFFF00", size=10),
                    textangle=-90 # Texto vertical para ocupar menos espacio
                )
        # --------------------------------------

        # Layout AUTOMÁTICO (Sin rangos fijos)
        fig_f.update_layout(
            template="plotly_dark", 
            height=350, 
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"Espectro de Magnitud ({f_unit})",
            xaxis=dict(
                title=f"Frecuencia ({f_unit})",
                type="log", # Mantenemos logarítmico porque es estándar en audio
                showgrid=True, 
                gridcolor='#333'
            ),
            yaxis=dict(
                title="Magnitud (dB)",
                # range=None,  <-- COMENTADO: Dejamos que Plotly decida
                showgrid=True, 
                gridcolor='#333'
            ),
            legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)'),
            uirevision=st.session_state.file_id # Mantiene el zoom manual del usuario si cambia sliders
        )
        st.plotly_chart(fig_f, use_container_width=True)

# --- OUTPUT ---
st.divider()
col_out1, col_out2 = st.columns([3, 1])
with col_out1:
    audio_out = np.nan_to_num(processed)
    pk = np.max(np.abs(audio_out))
    if pk > 0: audio_out /= pk
    audio_out = np.clip(audio_out, -1.0, 1.0)
    
    buffer = io.BytesIO()
    write(buffer, fs_out, (audio_out * 32767).astype(np.int16))
    render_player(buffer, fs_out, st.session_state.file_id)

with col_out2:
    st.download_button("💾 Descargar WAV", buffer, "dsp_out.wav", "audio/wav")

del resampled, processed, audio_out, buffer
gc.collect()
