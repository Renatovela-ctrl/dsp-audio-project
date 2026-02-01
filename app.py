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

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="DSP Workbench", layout="wide", page_icon="🎛️")

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

# --- 4. UTILIDADES ---
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
st.title("🎛️ Conversor de Frecuencia de Muestreo y Ecualizador para Señales de Audio en Tiempo Discreto - Israel Méndez, Daniel Molina, Renato Vela")

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

# LOOP
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

st.sidebar.subheader("Banco de Filtros")
bands = ["Sub", "Bass", "LoMid", "HiMid", "Pres", "Brill"]
keys = ["Sub-Bass", "Bass", "Low Mids", "High Mids", "Presence", "Brilliance"]
gains = {}
cols = st.sidebar.columns(3)
for i, (label, k) in enumerate(zip(bands, keys)):
    with cols[i%3]:
        gains[k] = st.slider(label, -15, 15, 0, key=f"eq_{i}")

# --- MOTOR DSP ---
resampled, fs_out = change_sampling_rate(work_data, fs_in, M, L)
processed = apply_equalizer(resampled, fs_out, gains)

# --- VISUALIZACIÓN ---
st.divider()
viz_mode = st.radio("Modo Visual:", ["🛠️ Análisis Completo", "📖 Teórico (50 muestras)"], horizontal=True)

if viz_mode == "🛠️ Análisis Completo":
    col_opt, _ = st.columns([1, 4])
    with col_opt:
        f_unit = st.radio("Eje X:", ["Hz", "rad/s"])
        x_mult = 2*np.pi if f_unit == "rad/s" else 1.0

    t1, t2 = st.tabs(["Tiempo", "Frecuencia"])
    
    # Tiempo
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
        # Frecuencia
        limit = min(len(work_data), 100000)
        fi, mi = compute_fft(work_data[:limit], fs_in)
        fo, mo = compute_fft(processed[:limit], fs_out)
        
        # Filtro de DC (0 Hz)
        mask_i = fi > 0.5 
        mask_o = fo > 0.5
        fi, mi = fi[mask_i], mi[mask_i]
        fo, mo = fo[mask_o], mo[mask_o]
        
        # dB
        mi_db = 20*np.log10(mi + 1e-9)
        mo_db = 20*np.log10(mo + 1e-9)
        
        # Downsample visual
        vi_f = safe_downsample(fi) * x_mult
        vi_m = safe_downsample(mi_db)
        vo_f = safe_downsample(fo) * x_mult
        vo_m = safe_downsample(mo_db)

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=vi_f, y=vi_m, name="In", line=dict(color='gray')))
        fig_f.add_trace(go.Scatter(x=vo_f, y=vo_m, name="Out", fill='tozeroy', line=dict(color='cyan')))
        
        # --- LÍNEAS VERTICALES CON ETIQUETAS ROTADAS ---
        boundaries = [60, 250, 2000, 4000, 6000]
        line_color = "#FF5500" # Naranja
        
        for b in boundaries:
            pos = b * x_mult
            # Línea vertical
            fig_f.add_vline(x=pos, line_dash="dot", line_color=line_color, opacity=0.8)
            
            # Etiqueta VERTICAL (-90 grados)
            # Esto evita que se extienda a la derecha
            if f_unit == "Hz":
                fig_f.add_annotation(
                    x=pos, 
                    y=0.95, yref="paper", # Parte superior
                    text=f"{b}", 
                    showarrow=False, 
                    font=dict(color=line_color, size=10),
                    textangle=-90,    # ROTACIÓN CLAVE
                    xanchor="left"    # Pegado a la línea
                )

        fig_f.update_layout(
            template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10),
            title=f"Espectro ({f_unit})", 
            xaxis=dict(type="log", title=f"Frecuencia ({f_unit})"),
            yaxis=dict(title="Magnitud (dB)"), 
            uirevision=st.session_state.file_id
        )
        st.plotly_chart(fig_f, use_container_width=True)

else:
    # MODO TEÓRICO
    st.markdown("#### 🔬 Análisis Discreto (Zoom 50 muestras)")
    n_samples = 40
    center = len(work_data) // 2
    
    slice_in = normalize_visuals(work_data[center : center+n_samples])
    
    ratio = fs_out / fs_in
    center_out = int(center * ratio)
    n_out = int(n_samples * ratio)
    slice_out = normalize_visuals(processed[center_out : center_out+n_out])

    fig_stem, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
    
    ax1.stem(range(len(slice_in)), slice_in, linefmt='k-', markerfmt='ko', basefmt='k-')
    ax1.set_title(r"Entrada $x[n]$", fontsize=10)
    ax1.grid(alpha=0.3)
    
    x_out_idx = np.linspace(0, len(slice_in), len(slice_out))
    ax2.stem(x_out_idx, slice_out, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax2.set_title(r"Salida $y[n]$", fontsize=10)
    ax2.grid(alpha=0.3)
    
    st.pyplot(fig_stem)

    st.markdown("#### 📐 Espectro Angular ($-\pi$ a $\pi$)")
    N_fft = 1024
    Win = np.fft.fftshift(np.fft.fft(work_data[:N_fft]))
    Wout = np.fft.fftshift(np.fft.fft(processed[:int(N_fft*ratio)]))
    
    w_axis_in = np.linspace(-np.pi, np.pi, len(Win))
    w_axis_out = np.linspace(-np.pi, np.pi, len(Wout))
    
    fig_w, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(w_axis_in, 20*np.log10(np.abs(Win)+1e-9), 'k--', alpha=0.5, label='Original')
    ax3.plot(w_axis_out, 20*np.log10(np.abs(Wout)+1e-9), 'r-', label='Procesada')
    ax3.set_xlim(-np.pi, np.pi)
    ax3.set_xlabel(r"Frecuencia $\omega$ (rad)")
    ax3.set_xticks([-np.pi, 0, np.pi])
    ax3.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax3.legend()
    ax3.grid(alpha=0.3)
    st.pyplot(fig_w)

# OUTPUT
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
