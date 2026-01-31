import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
import os
import gc
import base64
import time # <--- Nuevo: Para generar IDs únicos por tiempo
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="DSP Live Studio", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; }
    .block-container { padding-top: 1rem; }
    /* Estilo limpio para el reproductor */
    .dsp-player-container {
        background-color: #f8f9fa; 
        padding: 12px; 
        border-radius: 8px; 
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- ESTADO DE SESIÓN ---
if 'last_upload' not in st.session_state:
    st.session_state.last_upload = None

# --- CACHÉ (SOLO PARA PROCESOS MATEMÁTICOS PESADOS) ---
# Nota: Hemos quitado el caché de carga de archivos (load_audio) para evitar el error.
@st.cache_data(show_spinner=False)
def cached_resampling(data, fs, m, l):
    # Este sí lo dejamos porque resamplear es lento
    return change_sampling_rate(data, fs, m, l)

def downsample_visuals(data, max_points=2000):
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

# --- REPRODUCTOR HTML ROBUSTO ---
def render_audio_player(audio_bytes, fs, song_name):
    """
    Inyecta el audio usando un ID aleatorio cada vez.
    Esto obliga al navegador a no reciclar el reproductor anterior.
    """
    b64 = base64.b64encode(audio_bytes.read()).decode()
    
    # TRUCO: Añadimos timestamp al ID para que sea SIEMPRE único
    unique_id = int(time.time() * 1000)
    player_id = f"audio_{unique_id}"
    
    # Storage key única por canción (para recordar el tiempo de cada una por separado)
    # Limpiamos el nombre para que sea una key válida de JS
    clean_name = "".join(x for x in song_name if x.isalnum())
    storage_key = f"time_pos_{clean_name}" 
    
    html = f"""
    <div class="dsp-player-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; font-size: 0.9em; color: #495057;">
                🎵 {song_name} <span style="font-weight: 400; color: #adb5bd;">| {fs} Hz</span>
            </span>
            <span style="font-size: 0.7em; background: #e9ecef; padding: 2px 6px; border-radius: 4px; color: #6c757d;">
                LIVE MONITOR
            </span>
        </div>
        <audio id="{player_id}" controls autoplay style="width: 100%; height: 32px;">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    </div>

    <script>
        (function() {{
            var audio = document.getElementById('{player_id}');
            var key = '{storage_key}';
            
            // 1. Restaurar posición al cargar
            audio.onloadedmetadata = function() {{
                var saved = sessionStorage.getItem(key);
                if (saved && saved !== "null") {{
                    var t = parseFloat(saved);
                    // Validar que el tiempo guardado tiene sentido para ESTA canción
                    if (t < audio.duration && t > 0) {{
                        console.log("Restaurando a " + t + "s para {song_name}");
                        audio.currentTime = t;
                    }} else {{
                        // Si el tiempo guardado es mayor a la duración (cambio de canción), resetear
                        sessionStorage.setItem(key, 0);
                    }}
                }}
                
                var p = audio.play();
                if (p !== undefined) {{ p.catch(e => console.log("Autoplay blocked")); }}
            }};
            
            // 2. Guardar posición constantemente
            audio.ontimeupdate = function() {{
                if (audio.currentTime > 0) sessionStorage.setItem(key, audio.currentTime);
            }};
        }})();
    </script>
    """
    # Usamos un contenedor vacío que se sobrescribe para forzar renderizado
    st.markdown(html, unsafe_allow_html=True)

# --- INTERFAZ PRINCIPAL ---
st.title("🎛️ DSP Live Studio")

# --- 1. SELECCIÓN DE FUENTE ---
st.sidebar.header("1. Fuente de Audio")
input_mode = st.sidebar.radio("Origen:", ["📂 Subir Archivo", "🎵 Ejemplos"], horizontal=True)

current_data = None
current_fs = 0
file_id = ""

# LÓGICA DE CARGA SIN CACHÉ (Directa)
if input_mode == "📂 Subir Archivo":
    uploaded_file = st.sidebar.file_uploader("Arrastra tu WAV aquí", type=["wav"])
    if uploaded_file:
        # Carga directa sin caché
        current_data, current_fs = load_audio(uploaded_file)
        file_id = uploaded_file.name
else:
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        files = [f for f in os.listdir(examples_dir) if f.endswith('.wav')]
        if files:
            # Selectbox normal
            selected = st.sidebar.selectbox("Selecciona pista:", files)
            path = os.path.join(examples_dir, selected)
            
            # Carga directa sin caché (FIX para el problema de "se queda en la primera")
            current_data, current_fs = load_audio(path)
            file_id = selected
    else:
        st.sidebar.error("⚠️ Carpeta 'examples' no encontrada.")

# --- VALIDACIÓN DE CARGA ---
if current_data is not None:
    # Mostramos info técnica para confirmar que el archivo cambió
    duracion = len(current_data) / current_fs
    st.sidebar.caption(f"✅ Archivo cargado: **{file_id}**") 
    st.sidebar.caption(f"⏱️ Duración: {duracion:.2f}s | Muestras: {len(current_data)}")

    # --- 2. LOOP SETTINGS ---
    st.sidebar.markdown("---")
    use_loop = st.sidebar.toggle("⚡ Modo Rápido (Loop 15s)", value=True, 
                               help="Recomendado: Procesa solo un fragmento para que los controles respondan al instante.")

    if use_loop:
        mid = len(current_data) // 2
        window = 15 * current_fs
        start = max(0, mid - (window // 2))
        end = min(len(current_data), start + window)
        
        # Validar que no nos salimos de rango
        if end > len(current_data): end = len(current_data)
        
        working_data = current_data[start:end]
        st.sidebar.info(f"✂️ Editando tramo central ({start/current_fs:.1f}s - {end/current_fs:.1f}s)")
    else:
        working_data = current_data
        st.sidebar.warning("⚠️ Modo Completo activo. Si el audio es largo, la app será lenta.")

    # --- 3. CONTROLES DSP ---
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    L = col1.number_input("Upsample (L)", 1, 8, 1)
    M = col2.number_input("Downsample (M)", 1, 8, 1)

    st.sidebar.subheader("Ecualizador")
    cols = st.sidebar.columns(3)
    gains = {}
    bands = ["Sub", "Bass", "LoMid", "HiMid", "Pres", "Brill"]
    
    for i, band in enumerate(bands):
        with cols[i % 3]:
            gains[band] = st.slider(band, -15, 15, 0, key=f"sl_{i}") # Key única

    # Mapping para el core
    dsp_gains = {
        "Sub-Bass": gains["Sub"], "Bass": gains["Bass"], "Low Mids": gains["LoMid"],
        "High Mids": gains["HiMid"], "Presence": gains["Pres"], "Brilliance": gains["Brill"]
    }

    # --- 4. PROCESAMIENTO ---
    # Resampling
    if use_loop:
        # En modo loop NO usamos caché para garantizar frescura inmediata
        resampled, new_fs = change_sampling_rate(working_data, current_fs, M, L)
    else:
        # En modo full usamos caché
        resampled, new_fs = cached_resampling(working_data, current_fs, M, L)
    
    # EQ
    processed = apply_equalizer(resampled, new_fs, dsp_gains)

    # --- 5. VISUALIZACIÓN ---
    col_viz, col_play = st.columns([3, 2])

    with col_viz:
        t1, t2 = st.tabs(["⏱️ Tiempo", "🌊 Frecuencia"])
        
        with t1:
            fig_t = go.Figure()
            vp = downsample_visuals(processed, 1500)
            tx = np.linspace(0, len(vp)/new_fs, len(vp))
            fig_t.add_trace(go.Scatter(y=vp, x=tx, line=dict(color='#00cc96', width=1.5)))
            
            # MAGIA DEL ZOOM: uirevision + key estática
            fig_t.update_layout(height=280, margin=dict(t=20,b=20,l=20,r=20), 
                              showlegend=False, uirevision="const_t")
            st.plotly_chart(fig_t, use_container_width=True, key="p_time")

        with t2:
            f, m = compute_fft(processed, new_fs)
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=downsample_visuals(f, 1500), 
                                     y=downsample_visuals(20*np.log10(m+1e-9), 1500), 
                                     fill='tozeroy'))
            
            # MAGIA DEL ZOOM
            fig_f.update_layout(height=280, margin=dict(t=20,b=20,l=20,r=20), 
                              xaxis_type="log", showlegend=False, uirevision="const_f")
            st.plotly_chart(fig_f, use_container_width=True, key="p_freq")

    with col_play:
        # Pipeline de Audio
        clean = np.nan_to_num(processed)
        mx = np.max(np.abs(clean))
        if mx > 0: clean /= mx
        clean = np.clip(clean, -1.0, 1.0)
        
        wav_io = io.BytesIO()
        write(wav_io, new_fs, (clean * 32760).astype(np.int16))
        wav_io.seek(0)
        
        # Renderizamos reproductor pasando el NOMBRE del archivo para el ID
        render_audio_player(wav_io, new_fs, file_id)
        
        st.markdown("---")
        st.download_button("⬇️ Descargar WAV", wav_io, f"processed_{file_id}", "audio/wav")

    # Limpieza agresiva
    del resampled, processed, clean, wav_io
    gc.collect()

else:
    st.info("👈 Esperando archivo...")
