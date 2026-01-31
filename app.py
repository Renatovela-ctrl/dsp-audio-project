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
    /* Estilo para el reproductor embebido */
    .custom-player {
        width: 100%;
        margin-top: 10px;
        border-radius: 8px;
        outline: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CACHÉ ---
@st.cache_data(show_spinner=False)
def cached_resampling(data, fs, m, l):
    return change_sampling_rate(data, fs, m, l)

def downsample_visuals(data, max_points=2000):
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

# --- REPRODUCTOR "MARKDOWN" (MÁS ESTABLE QUE COMPONENTS) ---
def render_persistent_audio(audio_bytes, fs, key_suffix=""):
    """
    Inyecta el audio directamente en el DOM usando Markdown HTML.
    Esto evita los problemas de reinicio de iframes de Streamlit.
    """
    b64 = base64.b64encode(audio_bytes.read()).decode()
    player_id = f"dsp_audio_{key_suffix}"
    
    html_code = f"""
    <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #dee2e6;">
        <div style="font-size: 0.8em; color: #6c757d; margin-bottom: 5px;">
            MONITOR DE SALIDA ({fs} Hz)
        </div>
        <audio id="{player_id}" class="custom-player" controls autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    </div>

    <script>
        (function() {{
            var audio = document.getElementById('{player_id}');
            var storageKey = 'dsp_time_pos';

            // 1. EVENTO DE CARGA DE METADATOS (Restaurar posición)
            audio.onloadedmetadata = function() {{
                var savedTime = sessionStorage.getItem(storageKey);
                if (savedTime && savedTime !== "null") {{
                    var t = parseFloat(savedTime);
                    // Solo saltar si es válido y no estamos al final
                    if (t < audio.duration && t > 0) {{
                        audio.currentTime = t;
                    }}
                }}
                
                // Forzar reproducción
                var playPromise = audio.play();
                if (playPromise !== undefined) {{
                    playPromise.catch(error => {{ console.log("Autoplay bloqueado (interactúa primero)"); }});
                }}
            }};

            // 2. EVENTO DE ACTUALIZACIÓN DE TIEMPO (Guardar posición)
            audio.ontimeupdate = function() {{
                if (audio.currentTime > 0) {{
                    sessionStorage.setItem(storageKey, audio.currentTime);
                }}
            }};
        }})();
    </script>
    """
    # Usamos st.markdown con unsafe_allow_html=True para inyectarlo directo
    st.markdown(html_code, unsafe_allow_html=True)

# --- INTERFAZ ---
st.title("🎛️ DSP Live Studio")

# --- SIDEBAR ---
st.sidebar.header("1. Fuente")
input_mode = st.sidebar.radio("Modo:", ["📂 Subir", "🎵 Ejemplo"], horizontal=True)

input_data = None 
if input_mode == "📂 Subir":
    uploaded_file = st.sidebar.file_uploader("Archivo WAV", type=["wav"])
    if uploaded_file: input_data = uploaded_file
else:
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        files = [f for f in os.listdir(examples_dir) if f.endswith('.wav')]
        if files:
            selected = st.sidebar.selectbox("Track:", files)
            input_data = os.path.join(examples_dir, selected)

if input_data is not None:
    full_data, original_fs = load_audio(input_data)
    
    # --- MODO LOOP (ESTABILIDAD) ---
    st.sidebar.markdown("---")
    use_loop = st.sidebar.toggle("⚡ Modo Live (Loop 15s)", value=True, 
                               help="Trabaja sobre un fragmento para que los cambios sean instantáneos.")

    if use_loop:
        # Lógica de Loop
        mid_point = len(full_data) // 2
        window = 15 * original_fs
        start = max(0, mid_point - (window // 2))
        end = min(len(full_data), start + window)
        working_data = full_data[start:end]
        st.sidebar.caption(f"💡 Editando segundos {start/original_fs:.1f} a {end/original_fs:.1f}")
    else:
        working_data = full_data
        st.sidebar.warning("⚠️ Modo completo: Los cambios pueden tardar unos segundos.")

    # --- CONTROLES DSP ---
    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    L = c1.number_input("Upsample (L)", 1, 8, 1)
    M = c2.number_input("Downsample (M)", 1, 8, 1)

    st.sidebar.subheader("Ecualizador")
    cols = st.sidebar.columns(3)
    gains = {}
    bands_info = [("Sub", "16-60"), ("Bass", "60-250"), ("LoMid", "250-2k"),
                  ("HiMid", "2k-4k"), ("Pres", "4k-6k"), ("Brill", "6k-16k")]
    
    for i, (name, freq) in enumerate(bands_info):
        with cols[i % 3]:
            # Key única para cada slider
            gains[name] = st.slider(name, -15, 15, 0, key=f"eq_{i}", help=f"{freq} Hz")

    # --- PROCESAMIENTO ---
    # Decisión de caché según el modo
    if use_loop:
        resampled_data, new_fs = change_sampling_rate(working_data, original_fs, M, L)
    else:
        resampled_data, new_fs = cached_resampling(working_data, original_fs, M, L)
    
    processed_data = apply_equalizer(resampled_data, new_fs, gains)

    # --- VISUALIZACIÓN ---
    col_viz, col_play = st.columns([3, 2])

    with col_viz:
        tab1, tab2 = st.tabs(["⏱️ Tiempo", "🌊 Frecuencia"])
        
        with tab1:
            fig_t = go.Figure()
            # Downsampling agresivo para visualización fluida
            v_proc = downsample_visuals(processed_data, 1500)
            t_axis = np.linspace(0, len(v_proc)/new_fs, len(v_proc))
            
            fig_t.add_trace(go.Scatter(y=v_proc, x=t_axis, name="Señal", line=dict(color='#00cc96', width=1.5)))
            
            # --- AQUÍ ESTÁ LA MAGIA DEL ZOOM (uirevision) ---
            fig_t.update_layout(
                height=280, 
                margin=dict(t=10,b=10,l=10,r=10), 
                showlegend=False,
                uirevision="constant_time_view" # Mantiene el zoom fijo al actualizar
            )
            st.plotly_chart(fig_t, use_container_width=True)

        with tab2:
            f_o, m_o = compute_fft(processed_data, new_fs)
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=downsample_visuals(f_o, 1500), y=downsample_visuals(20*np.log10(m_o+1e-9), 1500), 
                                     name="Espectro", fill='tozeroy'))
            
            # --- AQUÍ ESTÁ LA MAGIA DEL ZOOM (uirevision) ---
            fig_f.update_layout(
                height=280, 
                margin=dict(t=10,b=10,l=10,r=10), 
                xaxis_type="log", 
                showlegend=False,
                uirevision="constant_freq_view" # Mantiene el zoom fijo al actualizar
            )
            st.plotly_chart(fig_f, use_container_width=True)

    with col_play:
        # AUDIO PIPELINE
        # 1. Limpieza rápida
        audio_clean = np.nan_to_num(processed_data)
        peak = np.max(np.abs(audio_clean))
        if peak > 0: audio_clean /= peak
        audio_clean = np.clip(audio_clean, -1.0, 1.0)
        
        # 2. BytesIO
        virtual_wav = io.BytesIO()
        write(virtual_wav, new_fs, (audio_clean * 32760).astype(np.int16))
        virtual_wav.seek(0)
        
        # 3. RENDERIZADO DEL REPRODUCTOR
        # Usamos la nueva función basada en Markdown (no st.audio, no components)
        render_persistent_audio(virtual_wav, new_fs)
        
        st.markdown("---")
        # Botón de descarga aparte
        st.download_button(
            label="⬇️ Descargar WAV Final",
            data=virtual_wav,
            file_name="dsp_output.wav",
            mime="audio/wav"
        )

    # --- GARBAGE COLLECTION ---
    del resampled_data, processed_data, audio_clean, virtual_wav
    gc.collect()

else:
    st.info("👈 Sube un archivo o usa un ejemplo.")
