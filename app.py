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
st.set_page_config(page_title="DSP Audio Lab", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; }
    .block-container { padding-top: 1rem; }
    /* Hacemos el reproductor HTML más bonito */
    audio { width: 100%; border-radius: 10px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- CACHÉ OPTIMIZADO ---
@st.cache_data(show_spinner=False)
def cached_resampling(data, fs, m, l):
    return change_sampling_rate(data, fs, m, l)

def downsample_visuals(data, max_points=2000):
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

# --- FUNCIÓN MÁGICA: REPRODUCTOR PERSISTENTE ---
def get_audio_player_html(audio_bytes, mime_type="audio/wav"):
    """
    Genera un reproductor HTML5 con JavaScript incrustado.
    El JS guarda la posición de reproducción en sessionStorage y la restaura
    automáticamente cuando la página se recarga (al mover un slider).
    """
    # 1. Convertir bytes a base64 para embeber en HTML
    b64 = base64.b64encode(audio_bytes.read()).decode()
    
    # 2. El ID único asegura que el navegador no confunda reproductores
    player_id = "dsp_audio_player"
    
    # 3. HTML + JS (El truco de la persistencia)
    html_code = f"""
    <audio id="{player_id}" controls autoplay>
        <source src="data:{mime_type};base64,{b64}" type="{mime_type}">
        Tu navegador no soporta el elemento de audio.
    </audio>
    <script>
        (function() {{
            var player = document.getElementById('{player_id}');
            
            // INTENTAR RESTAURAR POSICIÓN
            var savedTime = sessionStorage.getItem('{player_id}_time');
            if (savedTime && savedTime !== "null") {{
                // Pequeña validación para no saltar al final
                if (parseFloat(savedTime) < player.duration - 0.5) {{
                    player.currentTime = parseFloat(savedTime);
                }}
            }}
            
            // INTENTAR REPRODUCIR (Autoplay)
            // Los navegadores modernos a veces bloquean esto si no hay interacción previa,
            // pero como el usuario ya movió un slider, debería funcionar.
            var playPromise = player.play();
            if (playPromise !== undefined) {{
                playPromise.then(_ => {{
                    // Reproducción automática iniciada
                }}).catch(error => {{
                    console.log("Autoplay bloqueado por el navegador (normal en primera carga)");
                }});
            }}
            
            // GUARDAR POSICIÓN CONTINUAMENTE (Cada vez que avanza el audio)
            player.ontimeupdate = function() {{
                sessionStorage.setItem('{player_id}_time', player.currentTime);
            }};
            
            // RESETEAR AL TERMINAR
            player.onended = function() {{
                sessionStorage.setItem('{player_id}_time', 0);
            }};
        }})();
    </script>
    """
    return html_code

# --- INTERFAZ ---
st.title("🎛️ DSP Smart Player: Edición en Tiempo Real")
st.markdown("**Equipo:** Renato Vela, Israel Méndez, Daniel Molina")

# --- SIDEBAR ---
st.sidebar.header("1. Fuente")
input_mode = st.sidebar.radio("Modo:", ["📂 Subir Archivo", "🎵 Archivo de Ejemplo"], horizontal=True)

input_data = None 
if input_mode == "📂 Subir Archivo":
    uploaded_file = st.sidebar.file_uploader("WAV", type=["wav"])
    if uploaded_file: input_data = uploaded_file
else:
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        files = [f for f in os.listdir(examples_dir) if f.endswith('.wav')]
        if files:
            selected = st.sidebar.selectbox("Track:", files)
            input_data = os.path.join(examples_dir, selected)

if input_data is not None:
    # Carga Inicial
    original_data, original_fs = load_audio(input_data)
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. DSP Controls")
    
    # Controles más compactos
    c_L, c_M = st.sidebar.columns(2)
    L = c_L.number_input("L (Up)", 1, 8, 1)
    M = c_M.number_input("M (Down)", 1, 8, 1)

    st.sidebar.subheader("Ecualizador (dB)")
    # Sliders del EQ
    gains = {}
    bands = ["Sub-Bass", "Bass", "Low Mids", "High Mids", "Presence", "Brilliance"]
    freqs = ["16-60", "60-250", "250-2k", "2k-4k", "4k-6k", "6k-16k"]
    
    # Layout de sliders en grid 2x3 para ahorrar espacio
    cols = st.sidebar.columns(2)
    for i, band in enumerate(bands):
        with cols[i % 2]:
            gains[band] = st.slider(f"{band[:4]}", -12, 12, 0, help=freqs[i])

    # --- PROCESAMIENTO (FULL AUDIO) ---
    # Nota: Procesamos todo el audio. Para evitar crashes, confiamos en el GC al final.
    # Si el audio es muy largo (>5 min), Python sufrirá, pero para canciones estándar va bien.
    
    with st.spinner("Procesando audio..."):
        # 1. Resampling
        resampled_data, new_fs = cached_resampling(original_data, original_fs, M, L)
        
        # 2. EQ
        processed_data = apply_equalizer(resampled_data, new_fs, gains)

    # --- VISUALIZACIÓN ---
    col_viz, col_play = st.columns([3, 2])

    with col_viz:
        # Pestañas de gráficas
        tab1, tab2 = st.tabs(["⏱️ Tiempo", "🌊 Frecuencia"])
        
        with tab1:
            limit = min(len(original_data), 150000) # Ver primeros segundos
            fig_t = go.Figure()
            v_proc = downsample_visuals(processed_data[:limit], 1500)
            t_axis = np.linspace(0, limit/original_fs, len(v_proc))
            
            fig_t.add_trace(go.Scatter(y=v_proc, x=t_axis, name="Out", line=dict(width=1)))
            fig_t.update_layout(height=250, margin=dict(t=10,b=10,l=10,r=10), title=None)
            st.plotly_chart(fig_t, use_container_width=True)
            
        with tab2:
            # FFT (Solo calculamos para mostrar, usando versión downsampled para velocidad)
            f_o, m_o = compute_fft(processed_data[:limit], new_fs) # FFT de la sección inicial
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=downsample_visuals(f_o, 1500), y=downsample_visuals(20*np.log10(m_o+1e-9), 1500), name="Out"))
            fig_f.update_layout(height=250, margin=dict(t=10,b=10,l=10,r=10), xaxis_type="log", title=None)
            st.plotly_chart(fig_f, use_container_width=True)

    with col_play:
        st.markdown(f"### 🎧 Monitor ({new_fs} Hz)")
        
        # PREPARACIÓN DE AUDIO
        # 1. Limpieza matemática
        audio_final = np.nan_to_num(processed_data)
        mx = np.max(np.abs(audio_final))
        if mx > 0: audio_final /= mx
        audio_final = np.clip(audio_final, -1.0, 1.0)
        
        # 2. Generar BytesIO
        virtual_wav = io.BytesIO()
        write(virtual_wav, new_fs, (audio_final * 32760).astype(np.int16))
        virtual_wav.seek(0)
        
        # 3. INYECTAR REPRODUCTOR INTELIGENTE
        # Esto reemplaza a st.audio() con nuestra versión HTML+JS
        html_player = get_audio_player_html(virtual_wav)
        st.components.v1.html(html_player, height=60)
        
        # Botón de descarga aparte (porque el reproductor HTML no siempre tiene botón de descarga fácil)
        st.download_button("⬇️ Guardar Resultado", virtual_wav, "resultado_dsp.wav", "audio/wav")

    # --- LIMPIEZA DE MEMORIA (CRÍTICO PARA NO CRASHEAR) ---
    del resampled_data
    del processed_data
    del audio_final
    del virtual_wav
    gc.collect()

else:
    st.info("👈 Carga un audio para empezar.")
