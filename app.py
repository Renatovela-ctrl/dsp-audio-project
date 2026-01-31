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

# --- 1. CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="DSP Live Studio", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; }
    .block-container { padding-top: 1rem; }
    .dsp-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO (SESSION STATE) ---
# Aquí guardamos los datos para que no se pierdan ni se mezclen al recargar
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'audio_fs' not in st.session_state:
    st.session_state.audio_fs = 0
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""

# --- 3. FUNCIONES DE CARGA (CALLBACKS) ---
def load_file_callback():
    """Esta función se ejecuta SOLO cuando subes un archivo nuevo"""
    if st.session_state.uploader is not None:
        data, fs = load_audio(st.session_state.uploader)
        st.session_state.audio_data = data
        st.session_state.audio_fs = fs
        st.session_state.file_name = st.session_state.uploader.name
        # Forzar limpieza de memoria
        gc.collect()

def load_example_callback():
    """Esta función se ejecuta SOLO cuando cambias la selección de ejemplo"""
    selected_file = st.session_state.example_selector
    examples_dir = "examples"
    file_path = os.path.join(examples_dir, selected_file)
    
    if os.path.exists(file_path):
        data, fs = load_audio(file_path)
        st.session_state.audio_data = data
        st.session_state.audio_fs = fs
        st.session_state.file_name = selected_file
        gc.collect()

# --- 4. FUNCIONES DSP Y UTILITARIOS ---
def downsample_visuals(data, max_points=2000):
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

def render_smart_player(audio_bytes, fs, unique_id):
    """
    Reproductor HTML/JS que maneja la persistencia del tiempo.
    unique_id: Debe cambiar si la canción cambia, pero mantenerse si solo movemos EQ.
    """
    b64 = base64.b64encode(audio_bytes.read()).decode()
    
    # ID del elemento HTML (fijo para que Streamlit no lo destruya al redibujar)
    html_id = "persistent_audio_player"
    
    # Clave para sessionStorage (varía según la canción para no mezclar tiempos)
    storage_key = f"time_{unique_id}"
    
    html = f"""
    <div class="dsp-card">
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <strong>Monitor de Salida ({fs} Hz)</strong>
            <span style="font-size:0.8em; color:#666;">Persistencia Activa</span>
        </div>
        <audio id="{html_id}" controls autoplay style="width: 100%;">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    </div>

    <script>
        (function() {{
            var audio = document.getElementById('{html_id}');
            var key = '{storage_key}';
            
            // Cuando los metadatos cargan (duración, etc.)
            audio.onloadedmetadata = function() {{
                var saved = sessionStorage.getItem(key);
                
                if (saved && saved !== "null") {{
                    var t = parseFloat(saved);
                    // Si el tiempo guardado es válido para ESTA canción, saltamos
                    if (t < audio.duration && t > 0) {{
                        // Pequeño margen para evitar glitches
                        if (Math.abs(audio.currentTime - t) > 0.5) {{
                            audio.currentTime = t;
                        }}
                    }} else {{
                        // Si el tiempo es mayor que la duración (cambio de canción), resetear
                        sessionStorage.setItem(key, 0);
                    }}
                }}
                
                var p = audio.play();
                if (p !== undefined) {{ p.catch(e => {{ console.log("Autoplay waiting interaction"); }}); }}
            }};
            
            // Guardar tiempo cada vez que avanza
            audio.ontimeupdate = function() {{
                if (audio.currentTime > 0) sessionStorage.setItem(key, audio.currentTime);
            }};
        }})();
    </script>
    """
    st.components.v1.html(html, height=100)

# --- 5. INTERFAZ DE USUARIO ---
st.title("🎛️ DSP Live Studio")

# --- BARRA LATERAL: SELECCIÓN ---
st.sidebar.header("1. Fuente")
input_type = st.sidebar.radio("Tipo:", ["Ejemplo", "Subir"], horizontal=True)

if input_type == "Subir":
    st.sidebar.file_uploader(
        "Archivo WAV", type=["wav"], 
        key="uploader", 
        on_change=load_file_callback # <--- AQUÍ ESTÁ LA MAGIA (Callback)
    )
else:
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        files = [f for f in os.listdir(examples_dir) if f.endswith('.wav')]
        if files:
            st.sidebar.selectbox(
                "Selecciona:", files, 
                key="example_selector", 
                on_change=load_example_callback # <--- AQUÍ ESTÁ LA MAGIA (Callback)
            )
        else:
            st.sidebar.error("No hay archivos en /examples")

# --- VERIFICACIÓN DE CARGA ---
if st.session_state.audio_data is None:
    st.info("👈 Por favor carga un archivo para comenzar.")
    st.stop() # Detiene la ejecución aquí si no hay datos

# Si llegamos aquí, hay datos cargados en Session State
full_data = st.session_state.audio_data
original_fs = st.session_state.audio_fs
current_file = st.session_state.file_name

st.sidebar.success(f"Track: {current_file} | {len(full_data)/original_fs:.1f}s")

# --- 6. CONTROLES DSP ---
st.sidebar.markdown("---")
use_loop = st.sidebar.toggle("⚡ Modo Loop (15s)", value=True)

# Lógica de recorte
if use_loop:
    mid = len(full_data) // 2
    window = 15 * original_fs
    start = max(0, mid - (window // 2))
    end = min(len(full_data), start + window)
    working_data = full_data[start:end]
else:
    working_data = full_data

st.sidebar.subheader("Resampling")
c1, c2 = st.sidebar.columns(2)
L = c1.number_input("Upsample (L)", 1, 8, 1)
M = c2.number_input("Downsample (M)", 1, 8, 1)

st.sidebar.subheader("Ecualizador")
# Usamos un formulario para agrupar sliders si quisiéramos, pero directos es más rápido
cols = st.sidebar.columns(3)
bands = ["Sub", "Bass", "LoMid", "HiMid", "Pres", "Brill"]
freqs = ["16-60", "60-250", "250-2k", "2k-4k", "4k-6k", "6k-16k"]
gains = {}

for i, band in enumerate(bands):
    with cols[i % 3]:
        # Key estática basada en el nombre de la banda
        gains[band] = st.slider(band, -15, 15, 0, key=f"band_{i}", help=f"{freqs[i]} Hz")

# Mapeo
dsp_gains = {
    "Sub-Bass": gains["Sub"], "Bass": gains["Bass"], "Low Mids": gains["LoMid"],
    "High Mids": gains["HiMid"], "Presence": gains["Pres"], "Brilliance": gains["Brill"]
}

# --- 7. PROCESAMIENTO ---
# No usamos caché aquí a propósito para el modo Loop (queremos respuesta instantánea)
# La eficiencia viene de usar 'working_data' que es pequeño.
resampled, new_fs = change_sampling_rate(working_data, original_fs, M, L)
processed = apply_equalizer(resampled, new_fs, dsp_gains)

# --- 8. VISUALIZACIÓN ---
col_viz, col_play = st.columns([3, 2])

with col_viz:
    t1, t2 = st.tabs(["Tiempo", "Frecuencia"])
    
    with t1:
        fig_t = go.Figure()
        vp = downsample_visuals(processed, 1500)
        tx = np.linspace(0, len(vp)/new_fs, len(vp))
        fig_t.add_trace(go.Scatter(y=vp, x=tx, line=dict(color='#00cc96', width=1.5)))
        
        # ZOOM FIX: uirevision usa el nombre del archivo. 
        # Si cambias de archivo (nombre cambia), zoom resetea. 
        # Si solo mueves EQ (nombre igual), zoom se queda.
        fig_t.update_layout(
            height=250, margin=dict(t=10,b=10,l=10,r=10), showlegend=False,
            uirevision=f"time_{current_file}" 
        )
        st.plotly_chart(fig_t, use_container_width=True, config={'displayModeBar': False})

    with t2:
        f, m = compute_fft(processed, new_fs)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=downsample_visuals(f, 1500), 
            y=downsample_visuals(20*np.log10(m+1e-9), 1500), 
            fill='tozeroy'
        ))
        
        # ZOOM FIX
        fig_f.update_layout(
            height=250, margin=dict(t=10,b=10,l=10,r=10), showlegend=False,
            xaxis_type="log",
            uirevision=f"freq_{current_file}"
        )
        st.plotly_chart(fig_f, use_container_width=True, config={'displayModeBar': False})

with col_play:
    # Preparar Audio
    clean = np.nan_to_num(processed)
    mx = np.max(np.abs(clean))
    if mx > 0: clean /= mx
    clean = np.clip(clean, -1.0, 1.0)
    
    wav_io = io.BytesIO()
    write(wav_io, new_fs, (clean * 32760).astype(np.int16))
    wav_io.seek(0)
    
    # Renderizar Player
    # Usamos current_file como ID único para el storage de tiempo
    render_smart_player(wav_io, new_fs, unique_id=current_file)
    
    st.markdown("---")
    st.download_button("⬇️ Descargar WAV", wav_io, f"dsp_{current_file}", "audio/wav")

# Limpieza final
del resampled, processed, clean, wav_io
