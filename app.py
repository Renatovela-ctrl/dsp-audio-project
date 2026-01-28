import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
from scipy.io.wavfile import write
# Asegúrate de que tu archivo modules/dsp_core.py existe y está correcto
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- FUNCIÓN AUXILIAR PARA OPTIMIZAR GRÁFICAS ---
def downsample_for_plotting(data, max_points=10000):
    """
    Si hay más de 'max_points', toma muestras equiespaciadas.
    Esto evita que el navegador colapse al graficar millones de puntos.
    """
    n = len(data)
    if n == 0: return np.array([]) # Manejo de array vacío
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

# --- LÓGICA PRINCIPAL ---
if uploaded_file is not None:
    # 1. Cargar audio
    original_data, original_fs = load_audio(uploaded_file)
    st.sidebar.success(f"Fs Original: {original_fs} Hz")

    # --- SECCIÓN 2: CONVERSIÓN DE TASA ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. Conversión de Tasa (Resampling)")
    
    col1, col2 = st.sidebar.columns(2)
    L = col1.number_input("Factor L (Expansión)", min_value=1, value=1, step=1)
    M = col2.number_input("Factor M (Decimación)", min_value=1, value=1, step=1)

    # --- SECCIÓN 3: ECUALIZADOR (6 BANDAS) ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. Ecualizador")
    st.sidebar.caption("Ajuste de ganancia (dB) por banda")
    
    gains = {}
    gains["Sub-Bass"] = st.sidebar.slider("Sub-Bass (16-60Hz)", -20, 20, 0)
    gains["Bass"] = st.sidebar.slider("Bass (60-250Hz)", -20, 20, 0)
    gains["Low Mids"] = st.sidebar.slider("Low Mids (250-2k)", -20, 20, 0)
    gains["High Mids"] = st.sidebar.slider("High Mids (2k-4k)", -20, 20, 0)
    gains["Presence"] = st.sidebar.slider("Presence (4k-6k)", -20, 20, 0)
    gains["Brilliance"] = st.sidebar.slider("Brilliance (6k-16k)", -20, 20, 0)

    # --- PROCESAMIENTO DSP ---
    
    # Paso 1: Resampling
    resampled_data, new_fs = change_sampling_rate(original_data, original_fs, M, L)
    st.write(f"### Frecuencia de Muestreo Resultante: **{new_fs} Hz**")
    
    # Paso 2: Ecualización
    processed_data = apply_equalizer(resampled_data, new_fs, gains)

    # --- VISUALIZACIÓN ---
    tab1, tab2 = st.tabs(["⏱️ Dominio del Tiempo", "🌊 Dominio de la Frecuencia"])

    # PESTAÑA 1: TIEMPO
    with tab1:
        st.subheader("Comparación en el Tiempo")
        
        # Graficamos un tramo representativo
        limit_view = min(len(original_data), 100000)
        
        fig_time = go.Figure()
        
        y_orig_plot = downsample_for_plotting(original_data[:limit_view])
        y_proc_plot = downsample_for_plotting(processed_data[:limit_view])
        
        # Eje de tiempo aproximado
        if len(y_proc_plot) > 0:
            x_axis = np.linspace(0, limit_view/new_fs, len(y_proc_plot))
            
            fig_time.add_trace(go.Scatter(x=x_axis, y=y_orig_plot, name="Original", opacity=0.5))
            fig_time.add_trace(go.Scatter(x=x_axis, y=y_proc_plot, name="Procesada"))
            
            fig_time.update_layout(title="Forma de onda (Tramo inicial reducido)", xaxis_title="Tiempo (s)", yaxis_title="Amplitud")
            # Corrección de Warning: use_container_width es True por defecto en versiones nuevas,
            # pero usamos el argumento explícito compatible.
            st.plotly_chart(fig_time, use_container_width=True)

        # REPRODUCTOR DE AUDIO BLINDADO (AQUÍ ESTABA EL ERROR)
        st.markdown("### 🎧 Escuchar Resultado")
        
        # 1. Limpieza de NaNs (Not a Number) y Infs (Infinitos)
        audio_safe = np.nan_to_num(processed_data)
        
        # 2. Normalización segura
        max_val = np.max(np.abs(audio_safe))
        if max_val > 0:
            audio_safe = audio_safe / max_val
        
        # 3. CLIPPING DURO: Asegurar que nada salga del rango [-1.0, 1.0]
        # Esto elimina el error "overflow encountered in multiply"
        audio_safe = np.clip(audio_safe, -1.0, 1.0)
        
        # 4. Conversión a Enteros de 16-bits
        # Ahora es seguro multiplicar porque garantizamos rango [-1, 1]
        virtual_file = io.BytesIO()
        wav_data = (audio_safe * 32767).astype(np.int16)
        write(virtual_file, new_fs, wav_data)
        
        st.audio(virtual_file, format='audio/wav')

    # PESTAÑA 2: FRECUENCIA
    with tab2:
        st.subheader("Espectro de Frecuencia (FFT)")
        
        # Calcular FFT
        freq_in, mag_in = compute_fft(original_data, original_fs)
        freq_out, mag_out = compute_fft(processed_data, new_fs)
        
        # Optimizar puntos
        f_in_plot = downsample_for_plotting(freq_in, 5000)
        m_in_plot = downsample_for_plotting(mag_in, 5000)
        
        f_out_plot = downsample_for_plotting(freq_out, 5000)
        m_out_plot = downsample_for_plotting(mag_out, 5000)

        fig_freq = go.Figure()
        
        # Convertir a dB con seguridad
        db_in = 20 * np.log10(m_in_plot + 1e-10)
        db_out = 20 * np.log10(m_out_plot + 1e-10)

        fig_freq.add_trace(go.Scatter(x=f_in_plot, y=db_in, name="Entrada Original"))
        fig_freq.add_trace(go.Scatter(x=f_out_plot, y=db_out, name="Salida Procesada", line=dict(color='orange')))
        
        fig_freq.update_layout(
            xaxis_title="Frecuencia (Hz)", 
            yaxis_title="Magnitud (dB)", 
            xaxis_type="log", 
            title="Comparación Espectral (Optimizada)"
        )
        st.plotly_chart(fig_freq, use_container_width=True)

else:
    st.info("👋 Sube un archivo .wav en la barra lateral para comenzar.")
