import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
import os
import gc
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="DSP Audio Lab", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; }
    .main .block-container { padding-top: 1rem; }
    /* Ajuste para que los sliders sean más compactos */
    .stSlider { padding-bottom: 0rem; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES CACHÉ ---
@st.cache_data(show_spinner=False)
def cached_resampling(data, fs, m, l):
    return change_sampling_rate(data, fs, m, l)

def downsample_visuals(data, max_points=2000):
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

# --- TÍTULO ---
st.title("🎛️ DSP: Laboratorio de Audio en Tiempo Real")

# --- BARRA LATERAL ---
st.sidebar.header("1. Fuente")
input_mode = st.sidebar.radio("Origen:", ["📂 Subir", "🎵 Ejemplo"], horizontal=True)

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
    full_data, fs = load_audio(input_data)
    
    # --- CONTROL MAESTRO DE RENDIMIENTO ---
    st.sidebar.markdown("---")
    st.sidebar.header("🚀 Rendimiento")
    
    # ESTO ES LO QUE EVITA EL CRASH
    use_loop = st.sidebar.toggle("⚡ Modo Rápido (Loop 10s)", value=True, help="Procesa solo 10 segundos para ajuste en tiempo real sin colgar la memoria.")
    
    # Lógica de recorte (Slicing)
    if use_loop:
        # Tomar solo 10 segundos (o menos si el audio es corto)
        samples_10s = 10 * fs
        if len(full_data) > samples_10s:
            # Tomamos un tramo central interesante, no solo el inicio (ej. del segundo 10 al 20)
            start_sec = 0
            if len(full_data) > 20 * fs: start_sec = 10 * fs # Empezar en seg 10 si es larga
            working_data = full_data[start_sec : start_sec + samples_10s]
            st.sidebar.info("⚡ Modo Loop Activo: Editando tramo de 10s")
        else:
            working_data = full_data
    else:
        working_data = full_data
        st.sidebar.warning("⚠️ Modo Completo: El procesamiento será más lento.")

    # --- CONTROLES DSP ---
    st.sidebar.markdown("---")
    col_l, col_m = st.sidebar.columns(2)
    L = col_l.number_input("Upsampling (L)", 1, 8, 1)
    M = col_m.number_input("Downsampling (M)", 1, 8, 1)

    st.sidebar.subheader("Ecualizador")
    # Usamos columnas para los sliders para ahorrar espacio y ver todo junto
    c1, c2, c3 = st.sidebar.columns(3)
    g_sb = c1.slider("Sub", -12, 12, 0)
    g_bs = c2.slider("Bass", -12, 12, 0)
    g_lm = c3.slider("LowMid", -12, 12, 0)
    
    c4, c5, c6 = st.sidebar.columns(3)
    g_hm = c4.slider("HiMid", -12, 12, 0)
    g_pr = c5.slider("Pres", -12, 12, 0)
    g_br = c6.slider("Brill", -12, 12, 0)
    
    gains = {"Sub-Bass": g_sb, "Bass": g_bs, "Low Mids": g_lm, 
             "High Mids": g_hm, "Presence": g_pr, "Brilliance": g_br}

    # --- MOTOR DSP ---
    # 1. Resampling
    # Solo cacheamos si NO estamos en modo loop (porque el modo loop es rápido de por sí)
    if use_loop:
        resampled, new_fs = change_sampling_rate(working_data, fs, M, L)
    else:
        resampled, new_fs = cached_resampling(working_data, fs, M, L)
    
    # 2. EQ
    processed = apply_equalizer(resampled, new_fs, gains)

    # --- VISUALIZACIÓN ---
    col_main, col_side = st.columns([3, 1])

    with col_main:
        tab1, tab2 = st.tabs(["📊 Señal en Tiempo", "🌊 Espectro (FFT)"])
        
        with tab1:
            # Gráfica ultraligera
            fig_t = go.Figure()
            # Mostramos max 2000 puntos para velocidad extrema
            v_orig = downsample_visuals(working_data, 2000)
            v_proc = downsample_visuals(processed, 2000)
            
            t_axis = np.linspace(0, len(v_orig)/fs, len(v_orig)) # Tiempo relativo
            
            fig_t.add_trace(go.Scatter(y=v_orig, x=t_axis, name="Original", opacity=0.4))
            fig_t.add_trace(go.Scatter(y=v_proc, x=t_axis, name="Procesada"))
            fig_t.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20), legend=dict(y=1, x=1))
            st.plotly_chart(fig_t, use_container_width=True)

        with tab2:
            # FFT solo cuando estamos en la pestaña (ahorra cálculo si no se ve)
            f_i, m_i = compute_fft(working_data, fs)
            f_o, m_o = compute_fft(processed, new_fs)
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=downsample_visuals(f_i), y=downsample_visuals(20*np.log10(m_i+1e-9)), name="In"))
            fig_f.add_trace(go.Scatter(x=downsample_visuals(f_o), y=downsample_visuals(20*np.log10(m_o+1e-9)), name="Out"))
            fig_f.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20), xaxis_type="log", yaxis_title="dB")
            st.plotly_chart(fig_f, use_container_width=True)

    with col_side:
        st.write(f"### 🎧 Monitor")
        st.caption(f"Salida: **{new_fs} Hz**")
        
        # PREPARAR AUDIO
        audio_safe = np.nan_to_num(processed)
        mx = np.max(np.abs(audio_safe))
        if mx > 0: audio_safe /= mx
        audio_safe = np.clip(audio_safe, -1.0, 1.0)
        
        virtual_wav = io.BytesIO()
        write(virtual_wav, new_fs, (audio_safe * 32760).astype(np.int16))
        
        # Reproducción Automática (Experimental)
        # autoplay=True intenta reproducir apenas carga, dando sensación de continuidad
        st.audio(virtual_wav, format='audio/wav', autoplay=True)
        
        if not use_loop:
            st.success("✅ Procesamiento completo.")
            st.download_button("⬇️ Descargar WAV", virtual_wav, "audio_dsp.wav", "audio/wav")
        else:
            st.info("💡 Desactiva el 'Modo Rápido' para descargar la canción completa.")

    # LIMPIEZA OBLIGATORIA
    del working_data, resampled, processed, audio_safe
    gc.collect()

else:
    st.info("👈 Comienza seleccionando un audio.")
