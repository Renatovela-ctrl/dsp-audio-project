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
st.set_page_config(page_title="DSP Live Studio", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; }
    .block-container { padding-top: 1rem; }
    .dsp-card { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO (SESSION STATE) ---
# Inicialización de variables
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
    st.session_state.audio_fs = 0
    st.session_state.file_name = ""  # <--- CORREGIDO: Nombre unificado

# Estado para las Figuras (Persistencia visual)
if 'fig_time' not in st.session_state: st.session_state.fig_time = None
if 'fig_freq' not in st.session_state: st.session_state.fig_freq = None

# Estado para el Zoom
if 'zoom_uid' not in st.session_state: st.session_state.zoom_uid = str(uuid.uuid4())

# --- 3. CALLBACKS ---
def reset_system():
    """Resetea figuras y zoom al cambiar de archivo"""
    st.session_state.fig_time = None
    st.session_state.fig_freq = None
    st.session_state.zoom_uid = str(uuid.uuid4()) # Forzar reset de cámara

def load_file_callback():
    if st.session_state.uploader:
        d, fs = load_audio(st.session_state.uploader)
        st.session_state.audio_data = d
        st.session_state.audio_fs = fs
        st.session_state.file_name = st.session_state.uploader.name
        reset_system()

def load_example_callback():
    fname = st.session_state.example_selector
    path = os.path.join("examples", fname)
    if os.path.exists(path):
        d, fs = load_audio(path)
        st.session_state.audio_data = d
        st.session_state.audio_fs = fs
        st.session_state.file_name = fname
        reset_system()

def trigger_zoom_reset():
    """Genera un nuevo ID para obligar a Plotly a resetear la vista"""
    st.session_state.zoom_uid = str(uuid.uuid4())

# --- 4. UTILITARIOS ---
def downsample_visuals(data, max_points=1500):
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

def render_smart_player(audio_bytes, fs, unique_id):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    html_id = "persistent_player"
    # Usamos el nombre del archivo en la clave para no mezclar tiempos entre canciones
    clean_id = "".join(x for x in str(unique_id) if x.isalnum())
    storage_key = f"t_{clean_id}"
    
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
                    // Restaurar solo si es válido para ESTA canción
                    if (t < a.duration && t > 0) {{
                         a.currentTime = t;
                    }} else {{
                        // Si cambiamos de canción y el tiempo guardado es mayor a la duración nueva
                        sessionStorage.setItem(k, 0);
                    }}
                }}
                a.play().catch(e=>console.log("Autoplay waiting"));
            }};
            
            a.ontimeupdate = function() {{ 
                if(a.currentTime>0) sessionStorage.setItem(k, a.currentTime); 
            }};
        }})();
    </script>
    """
    st.components.v1.html(html, height=100)

# --- 5. INTERFAZ ---
st.title("🎛️ DSP Live Studio")

# BARRA LATERAL
st.sidebar.header("1. Fuente")
mode = st.sidebar.radio("Tipo:", ["Ejemplo", "Subir"], horizontal=True)

if mode == "Subir":
    st.sidebar.file_uploader("WAV", type=["wav"], key="uploader", on_change=load_file_callback)
else:
    if os.path.exists("examples"):
        files = [f for f in os.listdir("examples") if f.endswith('.wav')]
        if files:
            st.sidebar.selectbox("Pista:", files, key="example_selector", on_change=load_example_callback)

# VALIDACIÓN
if st.session_state.audio_data is None:
    st.info("👈 Carga un archivo para empezar.")
    st.stop()

# RECUPERAR DATOS (Ahora sí con el nombre correcto)
full_data = st.session_state.audio_data
original_fs = st.session_state.audio_fs
fname = st.session_state.file_name

st.sidebar.caption(f"Track: **{fname}** | {original_fs} Hz")

# --- CONFIGURACIÓN LOOP ---
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

# --- PROCESAMIENTO ---
resampled, new_fs = change_sampling_rate(working_data, original_fs, M, L)
processed = apply_equalizer(resampled, new_fs, dsp_gains)

# --- VISUALIZACIÓN ---
col_viz, col_play = st.columns([3, 2])

with col_viz:
    # Header con botón Reset
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        t1, t2 = st.tabs(["Tiempo (Orig vs Proc)", "Frecuencia"])
    with head_col2:
        st.write("") # Espaciador
        st.button("🔄 Reset Zoom", on_click=trigger_zoom_reset)

    # --- PESTAÑA TIEMPO ---
    with t1:
        # Datos Originales (Gris)
        v_orig = downsample_visuals(working_data, 1500)
        t_orig = np.linspace(0, len(v_orig)/original_fs, len(v_orig))
        
        # Datos Procesados (Verde)
        v_proc = downsample_visuals(processed, 1500)
        t_proc = np.linspace(0, len(v_proc)/new_fs, len(v_proc))

        # Crear o Actualizar Figura STATEFUL
        if st.session_state.fig_time is None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_orig, y=v_orig, name="Original", 
                                   line=dict(color='gray', width=1), opacity=0.4))
            fig.add_trace(go.Scatter(x=t_proc, y=v_proc, name="Procesada", 
                                   line=dict(color='#00cc96', width=1.5)))
            
            fig.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10), 
                            legend=dict(orientation="h", y=1.1, x=0))
            st.session_state.fig_time = fig
        else:
            # Solo inyectar datos, NO tocar layout (mantiene zoom)
            st.session_state.fig_time.data[0].x = t_orig
            st.session_state.fig_time.data[0].y = v_orig
            st.session_state.fig_time.data[1].x = t_proc
            st.session_state.fig_time.data[1].y = v_proc

        # Aplicar el ID de Zoom actual
        st.session_state.fig_time.layout.uirevision = st.session_state.zoom_uid
        st.plotly_chart(st.session_state.fig_time, use_container_width=True, key="chart_time")

    # --- PESTAÑA FRECUENCIA ---
    with t2:
        # FFT Orig
        fo, mo = compute_fft(working_data, original_fs)
        vf_o = downsample_visuals(fo, 1500)
        vm_o = downsample_visuals(20*np.log10(mo+1e-9), 1500)
        
        # FFT Proc
        fp, mp = compute_fft(processed, new_fs)
        vf_p = downsample_visuals(fp, 1500)
        vm_p = downsample_visuals(20*np.log10(mp+1e-9), 1500)

        if st.session_state.fig_freq is None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vf_o, y=vm_o, name="Original", 
                                   line=dict(color='gray', width=1), opacity=0.4))
            fig.add_trace(go.Scatter(x=vf_p, y=vm_p, name="Procesada", 
                                   fill='tozeroy', line=dict(color='#ef553b', width=1)))
            
            fig.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10), 
                            xaxis_type="log", legend=dict(orientation="h", y=1.1, x=0))
            st.session_state.fig_freq = fig
        else:
            st.session_state.fig_freq.data[0].x = vf_o
            st.session_state.fig_freq.data[0].y = vm_o
            st.session_state.fig_freq.data[1].x = vf_p
            st.session_state.fig_freq.data[1].y = vm_p

        st.session_state.fig_freq.layout.uirevision = st.session_state.zoom_uid
        st.plotly_chart(st.session_state.fig_freq, use_container_width=True, key="chart_freq")

# --- AUDIO ---
with col_play:
    # Preparar Audio
    clean = np.nan_to_num(processed)
    mx = np.max(np.abs(clean))
    if mx > 0: clean /= mx
    clean = np.clip(clean, -1.0, 1.0)
    
    wav_io = io.BytesIO()
    write(wav_io, new_fs, (clean * 32760).astype(np.int16))
    wav_io.seek(0)
    
    render_smart_player(wav_io, new_fs, fname)
    
    st.markdown("---")
    st.download_button("⬇️ Descargar WAV", wav_io, f"dsp_{fname}", "audio/wav")

gc.collect()
