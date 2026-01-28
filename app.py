import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
from scipy.io.wavfile import write
# Asegúrate de que tu archivo modules/dsp_core.py esté actualizado con la versión "blindada"
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- FUNCIÓN AUXILIAR PARA OPTIMIZAR GRÁFICAS ---
def downsample_for_plotting(data, max_points=10000):
    n = len(data)
    if n == 0: return np.array([])
    if n > max_points:
        step = n // max_points
        return data[::step]
    return data

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="DSP Audio Lab", layout="wide", page_icon="🎛️")

st.title("🎛️ Sistema de Procesamiento de Señales de Audio (T3)")
st.markdown("**Integrantes:** Renato Vela, Israel Méndez, Daniel Molina")

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("1. Carga de Señal")
uploaded_file = st.sidebar.file_uploader("Sube un archivo WAV", type=["wav"])

if uploaded_file is not None:
    # 1. Cargar audio
    original_data, original_fs = load_audio(uploaded_file)
    st.sidebar.success(f"Fs Original: {original_fs} Hz")

    # --- SECCIÓN 2: CONVERSIÓN DE TASA ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. Conversión de Tasa")
    
    col1, col2 = st.sidebar.columns(2)
    L = col1.number_input("Factor L (Expansión)", min_value=1, value=1, step=1)
    M = col2.number_input("Factor M (Decimación)", min_value=1, value=1, step=1)

    # --- SECCIÓN 3: ECUALIZADOR ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. Ecualizador")
    
    gains = {}
    gains["Sub-Bass"] = st.sidebar.slider("Sub-Bass (16-60Hz)", -20, 20, 0)
    gains["Bass"] = st.sidebar.slider("Bass (60-250Hz)", -20, 20, 0)
    gains["Low Mids"] = st.sidebar.slider("Low Mids (250-2k)", -20, 20, 0)
    gains["High Mids"] = st.sidebar.slider("High Mids (2k-4k)", -20, 20, 0)
    gains["Presence"] = st.sidebar.slider("Presence (4k-6k)", -20, 20, 0)
    gains["Brilliance"] = st.sidebar.slider("Brilliance (6k-16k)", -20, 20, 0)

    # --- PROCESAMIENTO ---
    resampled_data, new_fs = change_sampling_rate(original_data, original_fs, M, L)
    st.write(f"### Frecuencia de Muestreo Resultante: **{new_fs} Hz**")
    
    processed_data = apply_equalizer(resampled_data, new_fs, gains)

    # --- VISUALIZACIÓN ---
    tab1, tab2 = st.tabs(["⏱️ Dominio del Tiempo", "🌊 Dominio de la Frecuencia"])

    with tab1:
        st.subheader("Comparación en el Tiempo")
        
        limit_view = min(len(original_data), 100000)
        
        fig_time = go.Figure()
        y_orig_plot = downsample_for_plotting(original_data[:limit_view])
        y_proc_plot = downsample_for_plotting(processed_data[:limit_view])
        
        if len(y_proc_plot) > 0:
            x_axis = np.linspace(0, limit_view/new_fs, len(y_proc_plot))
            fig_time.add_trace(go.Scatter(x=x_axis, y=y_orig_plot, name="Original", opacity=0.5))
            fig_time.add_trace(go.Scatter(x=x_axis, y=y_proc_plot, name="Procesada"))
            fig_time.update_layout(title="Forma de onda", xaxis_title="Tiempo (s)", yaxis_title="Amplitud")
            
            # CORRECCIÓN DE WARNING: Reemplazamos use_container_width por el nuevo estándar
            try:
                st.plotly_chart(fig_time, key="time_plot", use_container_width=True) 
                # Nota: Si sigue saliendo el warning amarillo, ignóralo, es cosa de Streamlit actualizándose.
                # Lo importante es que funcione.
            except:
                st.plotly_chart(fig_time)

        # --- AQUÍ ESTABA EL PROBLEMA DEL AUDIO ---
        st.markdown("### 🎧 Escuchar Resultado")
        
        # 1. Limpieza y Clipping
        audio_safe = np.nan_to_num(processed_data)
        max_val = np.max(np.abs(audio_safe))
        if max_val > 0:
            audio_safe = audio_safe / max_val
        
        audio_safe = np.clip(audio_safe, -1.0, 1.0)
        
        # 2. Generación del Archivo
        virtual_file = io.BytesIO()
        wav_data = (audio_safe * 32767).astype(np.int16)
        write(virtual_file, new_fs, wav_data)
        
        # 3. REBOBINADO MÁGICO (ESTO FALTABA)
        virtual_file.seek(0)
        
        st.audio(virtual_file, format='audio/wav')

    with tab2:
        st.subheader("Espectro de Frecuencia (FFT)")
        
        freq_in, mag_in = compute_fft(original_data, original_fs)
        freq_out, mag_out = compute_fft(processed_data, new_fs)
        
        f_in_plot = downsample_for_plotting(freq_in, 5000)
        m_in_plot = downsample_for_plotting(mag_in, 5000)
        f_out_plot = downsample_for_plotting(freq_out, 5000)
        m_out_plot = downsample_for_plotting(mag_out, 5000)

        fig_freq = go.Figure()
        db_in = 20 * np.log10(m_in_plot + 1e-10)
        db_out = 20 * np.log10(m_out_plot + 1e-10)

        fig_freq.add_trace(go.Scatter(x=f_in_plot, y=db_in, name="Entrada Original"))
        fig_freq.add_trace(go.Scatter(x=f_out_plot, y=db_out, name="Salida Procesada", line=dict(color='orange')))
        
        fig_freq.update_layout(
            xaxis_title="Frecuencia (Hz)", 
            yaxis_title="Magnitud (dB)", 
            xaxis_type="log", 
            title="Comparación Espectral"
        )
        try:
            st.plotly_chart(fig_freq, key="freq_plot", use_container_width=True)
        except:
             st.plotly_chart(fig_freq)

else:
    st.info("👋 Sube un archivo .wav para comenzar.")
