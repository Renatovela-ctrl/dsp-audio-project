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

# --- REPRODUCTOR PERSISTENTE (CON MEMORIA) ---
def get_persistent_player_html(audio_bytes, fs):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    player_id = "dsp_player_v3"
    
    html = f"""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span style="font-weight: 600; font-size: 0.85em; color: #495057;">MONITOR ({fs} Hz)</span>
            <span style="font-size: 0.7em; color: #adb5bd;">Estado: Activo</span>
        </div>
        <audio id="{player_id}" controls autoplay style="width: 100%; height: 30px;">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    </div>

    <script>
        (function() {{
            var audio = document.getElementById('{player_id}');
            var storageKey = 'dsp_audio_timer_v3';

            // 1. Restaurar tiempo al cargar metadatos
            audio.onloadedmetadata = function() {{
                var savedTime = sessionStorage.getItem(storageKey);
                if (savedTime && savedTime !== "null") {{
                    var time = parseFloat(savedTime);
                    if (time < audio.duration && time > 0) {{
                        // Pequeño ajuste para evitar saltos bruscos
                        audio.currentTime = time; 
                    }}
                }}
                
                var playPromise = audio.play();
                if (playPromise !== undefined) {{
                    playPromise.catch(error => {{ console.log("Autoplay esperando..."); }});
                }}
            }};

            // 2. Guardar tiempo continuamente
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
    
    # --- MODO LOOP ---
    st.sidebar.markdown("---")
    use_loop = st.sidebar.toggle("⚡ Modo Live (Loop 15s)", value=True)

    if use_loop:
        mid_point = len(full_data) // 2
        window = 15 * original_fs
        start = max(0, mid_point - (window // 2))
        end = min(len(full_data), start + window)
        working_data = full_data[start:end]
        st.sidebar.caption("💡 Editando tramo central para respuesta inmediata.")
    else:
        working_data = full_data
        st.sidebar.warning("⚠️ Modo completo: Puede ser lento.")

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
            gains[name] = st.slider(name, -15, 15, 0, key=f"eq_{i}", help=f"{freq} Hz")

    # --- PROCESAMIENTO ---
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
            v_proc = downsample_visuals(processed_data, 1500)
            t_axis = np.linspace(0, len(v_proc)/new_fs, len(v_proc))
            
            fig_t.add_trace(go.Scatter(y=v_proc, x=t_axis, name="Señal", line=dict(color='#00cc96', width=1.5)))
            
            # MAGIA PARA MANTENER ZOOM: uirevision="constant"
            fig_t.update_layout(
                height=280, 
                margin=dict(t=10,b=10,l=10,r=10), 
                showlegend=False,
                uirevision="constant" # <--- ESTO MANTIENE EL ZOOM FIJO
            )
            st.plotly_chart(fig_t, use_container_width=True)

        with tab2:
            f_o, m_o = compute_fft(processed_data, new_fs)
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=downsample_visuals(f_o, 1500), y=downsample_visuals(20*np.log10(m_o+1e-9), 1500), 
                                     name="Espectro", fill='tozeroy'))
            
            # MAGIA PARA MANTENER ZOOM: uirevision="constant"
            fig_f.update_layout(
                height=280, 
                margin=dict(t=10,b=10,l=10,r=10), 
                xaxis_type="log", 
                showlegend=False,
                uirevision="constant" # <--- ESTO MANTIENE EL ZOOM FIJO
            )
            st.plotly_chart(fig_f, use_container_width=True)

    with col_play:
        # AUDIO PIPELINE
        audio_clean = np.nan_to_num(processed_data)
        peak = np.max(np.abs(audio_clean))
        if peak > 0: audio_clean /= peak
        audio_clean = np.clip(audio_clean, -1.0, 1.0)
        
        virtual_wav = io.BytesIO()
        write(virtual_wav, new_fs, (audio_clean * 32760).astype(np.int16))
        virtual_wav.seek(0)
        
        # Player HTML
        html_code = get_persistent_player_html(virtual_wav, new_fs)
        st.components.v1.html(html_code, height=90)
        
        st.markdown("---")
        st.download_button("⬇️ Descargar WAV", virtual_wav, "resultado.wav", "audio/wav")

    del resampled_data, processed_data, audio_clean, virtual_wav
    gc.collect()

else:
    st.info("👈 Sube un archivo o usa un ejemplo.")
