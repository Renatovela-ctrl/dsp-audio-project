import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
import os
import gc
import base64
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="DSP Live Studio", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; }
    .block-container { padding-top: 1rem; }
    /* Clase para el reproductor */
    .dsp-player { width: 100%; margin-top: 10px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- GESTIÓN DE ESTADO (SESSION STATE) ---
if 'last_file' not in st.session_state:
    st.session_state.last_file = None

# --- CACHÉ ---
@st.cache_data(show_spinner=False)
def cached_load(file_path_or_buffer):
    # Cacheamos la carga para no leer el disco cada vez que mueves un slider
    return load_audio(file_path_or_buffer)

@st.cache_data(show_spinner=False)
def cached_resampling(data, fs, m, l):
    return change_sampling_rate(data, fs, m, l)

def downsample_visuals(data, max_points=2000):
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

# --- REPRODUCTOR ROBUSTO ---
def render_audio_player(audio_bytes, fs, song_id):
    """
    song_id: Es CRÍTICO. Si cambiamos de canción, este ID cambia, 
    obligando al navegador a crear un reproductor nuevo y olvidar el anterior.
    """
    b64 = base64.b64encode(audio_bytes.read()).decode()
    # Usamos el song_id para que el navegador sepa que es OTRO archivo
    player_id = f"player_{song_id}" 
    storage_key = f"time_{song_id}" # Guardamos el tiempo POR CANCIÓN
    
    html = f"""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6;">
        <div style="font-size: 0.8em; color: #6c757d;">MONITOR ({fs} Hz) - ID: {song_id}</div>
        <audio id="{player_id}" class="dsp-player" controls autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    </div>
    <script>
        (function() {{
            var audio = document.getElementById('{player_id}');
            var key = '{storage_key}';
            
            audio.onloadedmetadata = function() {{
                var saved = sessionStorage.getItem(key);
                if (saved && saved !== "null") {{
                    var t = parseFloat(saved);
                    if (t < audio.duration && t > 0) audio.currentTime = t;
                }}
                audio.play().catch(e => console.log("Autoplay esperando..."));
            }};
            
            audio.ontimeupdate = function() {{
                if (audio.currentTime > 0) sessionStorage.setItem(key, audio.currentTime);
            }};
        }})();
    </script>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- INTERFAZ ---
st.title("🎛️ DSP Live Studio")

# --- 1. SELECCIÓN DE FUENTE (FIXED) ---
st.sidebar.header("1. Fuente")
input_mode = st.sidebar.radio("Modo:", ["📂 Subir", "🎵 Ejemplo"], horizontal=True)

current_data = None
current_fs = 0
file_identifier = "none" # Esto servirá para identificar la canción única

if input_mode == "📂 Subir":
    uploaded_file = st.sidebar.file_uploader("WAV", type=["wav"])
    if uploaded_file:
        current_data, current_fs = cached_load(uploaded_file)
        file_identifier = uploaded_file.name
else:
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        files = [f for f in os.listdir(examples_dir) if f.endswith('.wav')]
        if files:
            # CORRECCIÓN: key='ex_select' evita conflictos de estado
            selected = st.sidebar.selectbox("Track:", files, key='ex_select')
            path = os.path.join(examples_dir, selected)
            current_data, current_fs = cached_load(path)
            file_identifier = selected # "piano.wav", etc.

# --- LÓGICA DE RESETEO DE SESIÓN ---
# Si cambiamos de canción, limpiamos variables viejas para evitar mezclas
if st.session_state.last_file != file_identifier:
    st.session_state.last_file = file_identifier
    # Forzamos recolección de basura al cambiar de canción
    gc.collect()

if current_data is not None:
    
    # --- 2. CONFIGURACIÓN DEL PROCESO ---
    st.sidebar.markdown("---")
    use_loop = st.sidebar.toggle("⚡ Modo Live (Loop 15s)", value=True)

    if use_loop:
        mid = len(current_data) // 2
        window = 15 * current_fs
        start = max(0, mid - (window // 2))
        end = min(len(current_data), start + window)
        working_data = current_data[start:end]
        st.sidebar.caption(f"✂️ Editando Loop: {start/current_fs:.1f}s - {end/current_fs:.1f}s")
    else:
        working_data = current_data
        st.sidebar.warning("⚠️ Modo Completo (Puede ser lento)")

    # --- 3. CONTROLES DSP ---
    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    L = c1.number_input("Upsample (L)", 1, 8, 1)
    M = c2.number_input("Downsample (M)", 1, 8, 1)

    st.sidebar.subheader("Ecualizador")
    cols = st.sidebar.columns(3)
    gains = {}
    bands = ["Sub", "Bass", "LoMid", "HiMid", "Pres", "Brill"]
    freqs = ["16-60", "60-250", "250-2k", "2k-4k", "4k-6k", "6k-16k"]
    
    for i, band in enumerate(bands):
        with cols[i % 3]:
            gains[band] = st.slider(band, -15, 15, 0, key=f"eq_{i}", help=f"{freqs[i]} Hz")
    
    # Mapeo de nombres cortos a nombres del DSP core
    gains_mapped = {
        "Sub-Bass": gains["Sub"], "Bass": gains["Bass"], "Low Mids": gains["LoMid"],
        "High Mids": gains["HiMid"], "Presence": gains["Pres"], "Brilliance": gains["Brill"]
    }

    # --- 4. MOTOR DSP ---
    if use_loop:
        # Loop: Sin caché para máxima respuesta
        resampled, new_fs = change_sampling_rate(working_data, current_fs, M, L)
    else:
        # Full: Con caché
        resampled, new_fs = cached_resampling(working_data, current_fs, M, L)
    
    processed = apply_equalizer(resampled, new_fs, gains_mapped)

    # --- 5. VISUALIZACIÓN ---
    col_viz, col_play = st.columns([3, 2])

    with col_viz:
        tab1, tab2 = st.tabs(["⏱️ Tiempo", "🌊 Frecuencia"])
        
        with tab1:
            fig_t = go.Figure()
            vp = downsample_visuals(processed, 1500)
            tx = np.linspace(0, len(vp)/new_fs, len(vp))
            fig_t.add_trace(go.Scatter(y=vp, x=tx, line=dict(color='#00cc96', width=1.5)))
            
            # SOLUCIÓN ZOOM: uirevision + KEY estática en plotly_chart
            fig_t.update_layout(height=280, margin=dict(t=20,b=20,l=20,r=20), 
                              uirevision="const_time") 
            st.plotly_chart(fig_t, use_container_width=True, key="plot_time")

        with tab2:
            f, m = compute_fft(processed, new_fs)
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=downsample_visuals(f, 1500), 
                                     y=downsample_visuals(20*np.log10(m+1e-9), 1500), 
                                     fill='tozeroy'))
            
            # SOLUCIÓN ZOOM: uirevision + KEY estática
            fig_f.update_layout(height=280, margin=dict(t=20,b=20,l=20,r=20), 
                              xaxis_type="log", uirevision="const_freq")
            st.plotly_chart(fig_f, use_container_width=True, key="plot_freq")

    with col_play:
        # Pipeline de Audio
        clean = np.nan_to_num(processed)
        mx = np.max(np.abs(clean))
        if mx > 0: clean /= mx
        clean = np.clip(clean, -1.0, 1.0)
        
        wav_io = io.BytesIO()
        write(wav_io, new_fs, (clean * 32760).astype(np.int16))
        wav_io.seek(0)
        
        # SOLUCIÓN CANCIONES: Pasamos 'file_identifier' para crear un ID único
        # Solo cambiará si cambias de archivo, no si cambias el EQ.
        render_audio_player(wav_io, new_fs, song_id=file_identifier)
        
        st.markdown("---")
        st.download_button("⬇️ Descargar WAV", wav_io, "dsp_result.wav", "audio/wav")

    # Limpieza
    del resampled, processed, clean, wav_io
    gc.collect()

else:
    st.info("👈 Selecciona una fuente de audio.")
