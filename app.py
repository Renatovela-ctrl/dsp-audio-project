import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
import os
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="DSP Audio Lab", layout="wide", page_icon="🎛️")

# Estilos CSS
st.markdown("""
    <style>
    .stAlert { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- CACHÉ ---
@st.cache_data
def cached_resampling(data, fs, m, l):
    return change_sampling_rate(data, fs, m, l)

def downsample_visuals(data, max_points=5000):
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

# --- INTERFAZ ---
st.title("🎛️ DSP: Conversor de Tasa y Ecualizador")
st.markdown("**Equipo:** Renato Vela, Israel Méndez, Daniel Molina")

# --- SIDEBAR (LÓGICA NUEVA: SUBIR O EJEMPLO) ---
st.sidebar.header("1. Entrada de Audio")

# Selector de modo
input_mode = st.sidebar.radio("Fuente:", ["📂 Subir Archivo", "🎵 Usar Ejemplo"])

input_data = None # Aquí guardaremos el archivo seleccionado

if input_mode == "📂 Subir Archivo":
    uploaded_file = st.sidebar.file_uploader("Sube un archivo WAV", type=["wav"])
    if uploaded_file is not None:
        input_data = uploaded_file

else: # Modo Ejemplo
    examples_dir = "examples"
    # Verificar si existe la carpeta
    if os.path.exists(examples_dir):
        # Listar archivos .wav
        files = [f for f in os.listdir(examples_dir) if f.endswith('.wav')]
        if files:
            selected_file = st.sidebar.selectbox("Elige un audio:", files)
            # Construir ruta completa
            input_data = os.path.join(examples_dir, selected_file)
        else:
            st.sidebar.warning("No hay archivos .wav en la carpeta 'examples'.")
    else:
        st.sidebar.error("Carpeta 'examples' no encontrada en el repositorio.")

# --- LÓGICA PRINCIPAL ---
if input_data is not None:
    # Carga Robustecida
    original_data, original_fs = load_audio(input_data)
    st.sidebar.success(f"Cargado: {original_fs} Hz")

    st.sidebar.markdown("---")
    st.sidebar.header("2. Resampling")
    col1, col2 = st.sidebar.columns(2)
    L = col1.number_input("Expansión (L)", 1, 10, 1)
    M = col2.number_input("Decimación (M)", 1, 10, 1)

    st.sidebar.markdown("---")
    st.sidebar.header("3. Ecualizador (dB)")
    gains = {
        "Sub-Bass": st.sidebar.slider("16-60Hz", -12, 12, 0),
        "Bass": st.sidebar.slider("60-250Hz", -12, 12, 0),
        "Low Mids": st.sidebar.slider("250-2k", -12, 12, 0),
        "High Mids": st.sidebar.slider("2k-4k", -12, 12, 0),
        "Presence": st.sidebar.slider("4k-6k", -12, 12, 0),
        "Brilliance": st.sidebar.slider("6k-16k", -12, 12, 0),
    }

    # --- PROCESAMIENTO ---
    resampled_data, new_fs = cached_resampling(original_data, original_fs, M, L)
    st.metric("Nueva Frecuencia de Muestreo", f"{new_fs} Hz")
    
    processed_data = apply_equalizer(resampled_data, new_fs, gains)

    # --- SALIDAS VISUALES ---
    tab_t, tab_f = st.tabs(["⏱️ Tiempo", "🌊 Frecuencia"])

    with tab_t:
        limit = min(len(original_data), 100000)
        fig = go.Figure()
        
        t_orig = np.linspace(0, limit/original_fs, num=len(downsample_visuals(original_data[:limit])))
        t_proc = np.linspace(0, limit/original_fs, num=len(downsample_visuals(processed_data[:limit])))
        
        fig.add_trace(go.Scatter(y=downsample_visuals(original_data[:limit]), x=t_orig, name="Original", opacity=0.5))
        fig.add_trace(go.Scatter(y=downsample_visuals(processed_data[:limit]), x=t_proc, name="Procesada"))
        fig.update_layout(title="Comparativa Temporal", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🎧 Resultado")
        audio_final = np.nan_to_num(processed_data)
        max_amp = np.max(np.abs(audio_final))
        if max_amp > 0: audio_final = audio_final / max_amp
        audio_final = np.clip(audio_final, -1.0, 1.0)
        
        virtual_wav = io.BytesIO()
        int_data = (audio_final * 32760).astype(np.int16)
        write(virtual_wav, new_fs, int_data)
        virtual_wav.seek(0)
        st.audio(virtual_wav, format='audio/wav')

    with tab_f:
        f_in, mag_in = compute_fft(original_data, original_fs)
        f_out, mag_out = compute_fft(processed_data, new_fs)
        
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=downsample_visuals(f_in), y=downsample_visuals(20*np.log10(mag_in+1e-9)), name="Entrada"))
        fig_fft.add_trace(go.Scatter(x=downsample_visuals(f_out), y=downsample_visuals(20*np.log10(mag_out+1e-9)), name="Salida"))
        
        fig_fft.update_layout(xaxis_type="log", title="Espectro de Magnitud (dB)", yaxis_title="dB", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_fft, use_container_width=True)

else:
    st.info("👈 Selecciona un archivo de ejemplo o sube uno propio en el menú lateral.")
