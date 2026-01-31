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

# CSS para limpiar la interfaz y ocultar errores
st.markdown("""
    <style>
    .stAlert { display: none; }
    .block-container { padding-top: 1rem; }
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

# --- EL REPRODUCTOR "PRESERVATIVO" (JAVASCRIPT CORREGIDO) ---
def get_persistent_player_html(audio_bytes, fs):
    """
    Genera un reproductor que espera a cargar los metadatos antes de saltar 
    al tiempo guardado. Usa sessionStorage para persistencia.
    """
    b64 = base64.b64encode(audio_bytes.read()).decode()
    player_id = "dsp_player_v2" # ID único
    
    html = f"""
    <div style="background-color: #f1f3f6; padding: 10px; border-radius: 10px; border: 1px solid #ddd;">
        <p style="margin: 0px 0px 5px 0px; font-weight: bold; font-size: 0.9em; color: #333;">
            Monitor de Salida ({fs} Hz)
        </p>
        <audio id="{player_id}" controls autoplay style="width: 100%;">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    </div>

    <script>
        (function() {{
            var audio = document.getElementById('{player_id}');
            var storageKey = 'dsp_audio_timer';

            // 1. AL CARGAR METADATOS (CRÍTICO: Aquí es donde fallaba antes)
            audio.onloadedmetadata = function() {{
                var savedTime = sessionStorage.getItem(storageKey);
                
                if (savedTime && savedTime !== "null") {{
                    var time = parseFloat(savedTime);
                    // Solo saltamos si el tiempo es válido y menor a la duración
                    if (time < audio.duration && time > 0) {{
                        console.log("Restaurando tiempo a: " + time);
                        audio.currentTime = time;
                    }}
                }}
                
                // Intentar reproducir (Autoplay policy puede requerir interacción previa)
                var playPromise = audio.play();
                if (playPromise !== undefined) {{
                    playPromise.catch(error => {{
                        console.log("Autoplay esperando interacción...");
                    }});
                }}
            }};

            // 2. GUARDAR TIEMPO MIENTRAS SUENA
            audio.ontimeupdate = function() {{
                if (audio.currentTime > 0) {{
                    sessionStorage.setItem(storageKey, audio.currentTime);
                }}
            }};
        }})();
    </script>
    """
    return html

# --- INTERFAZ ---
st.title("🎛️ DSP Live Studio (Renato, Israel, Daniel)")

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
    # Carga Inicial
    full_data, original_fs = load_audio(input_data)
    
    # --- MODO LOOP (Mantiene la estabilidad) ---
    st.sidebar.markdown("---")
    # Por defecto activado para evitar CRASHES al mover sliders rápido
    use_loop = st.sidebar.toggle("⚡ Modo Live (Loop 15s)", value=True)

    if use_loop:
        # Cortamos 15 segundos del medio para probar
        mid_point = len(full_data) // 2
        window = 15 * original_fs
        start = max(0, mid_point - (window // 2))
        end = min(len(full_data), start + window)
        working_data = full_data[start:end]
        st.sidebar.caption("💡 Editando segmento central de 15s para respuesta instantánea.")
    else:
        working_data = full_data
        st.sidebar.warning("⚠️ Procesando archivo completo. Puede ser lento.")

    # --- CONTROLES DSP ---
    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    L = c1.number_input("L (Upsample)", 1, 8, 1)
    M = c2.number_input("M (Downsample)", 1, 8, 1)

    st.sidebar.subheader("Ecualizador Gráfico")
    # Sliders directos (sin st.form para ver cambios en vivo)
    cols = st.sidebar.columns(3)
    gains = {}
    bands_info = [
        ("Sub-Bass", "16-60"), ("Bass", "60-250"), ("Low Mids", "250-2k"),
        ("High Mids", "2k-4k"), ("Presence", "4k-6k"), ("Brilliance", "6k-16k")
    ]
    
    for i, (name, freq) in enumerate(bands_info):
        with cols[i % 3]:
            gains[name] = st.slider(name, -15, 15, 0, key=f"eq_{i}", help=f"{freq} Hz")

    # --- PROCESAMIENTO ---
    # Resampling
    if use_loop:
        # Sin caché en loop para máxima velocidad
        resampled_data, new_fs = change_sampling_rate(working_data, original_fs, M, L)
    else:
        # Con caché en modo completo
        resampled_data, new_fs = cached_resampling(working_data, original_fs, M, L)
    
    # Ecualización
    processed_data = apply_equalizer(resampled_data, new_fs, gains)

    # --- VISUALIZACIÓN ---
    col_viz, col_play = st.columns([3, 2])

    with col_viz:
        tab1, tab2 = st.tabs(["⏱️ Tiempo", "🌊 Frecuencia"])
        
        with tab1:
            # Gráfica muy ligera (Max 1500 puntos)
            fig_t = go.Figure()
            v_proc = downsample_visuals(processed_data, 1500)
            t_axis = np.linspace(0, len(v_proc)/new_fs, len(v_proc))
            
            fig_t.add_trace(go.Scatter(y=v_proc, x=t_axis, name="Señal", line=dict(color='#00cc96', width=1.5)))
            fig_t.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)

        with tab2:
            # FFT (Solo calculamos lo necesario)
            f_o, m_o = compute_fft(processed_data, new_fs)
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=downsample_visuals(f_o, 1500), y=downsample_visuals(20*np.log10(m_o+1e-9), 1500), 
                                     name="Espectro", fill='tozeroy'))
            fig_f.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10), xaxis_type="log", showlegend=False)
            st.plotly_chart(fig_f, use_container_width=True)

    with col_play:
        # PREPARACIÓN SEGURA DEL AUDIO
        # 1. Limpieza (NaNs a 0)
        audio_clean = np.nan_to_num(processed_data)
        
        # 2. Normalización suave (evita saturación digital)
        peak = np.max(np.abs(audio_clean))
        if peak > 0: audio_clean /= peak
        
        # 3. Clip de seguridad
        audio_clean = np.clip(audio_clean, -1.0, 1.0)
        
        # 4. Generar archivo en RAM
        virtual_wav = io.BytesIO()
        write(virtual_wav, new_fs, (audio_clean * 32760).astype(np.int16))
        virtual_wav.seek(0)
        
        # 5. RENDERIZAR REPRODUCTOR INTELIGENTE
        # Usamos components.html para inyectar el script con seguridad
        html_code = get_persistent_player_html(virtual_wav, new_fs)
        st.components.v1.html(html_code, height=100)
        
        # Botón de descarga
        st.markdown("---")
        st.download_button(
            label="⬇️ Descargar WAV Procesado",
            data=virtual_wav,
            file_name="dsp_output.wav",
            mime="audio/wav"
        )

    # --- GARBAGE COLLECTION ---
    del resampled_data, processed_data, audio_clean, virtual_wav
    gc.collect()

else:
    st.info("👈 Sube un archivo o usa un ejemplo para comenzar.")
