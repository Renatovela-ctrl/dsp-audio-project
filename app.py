import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- FUNCIÓN AUXILIAR PARA NO MATAR AL NAVEGADOR ---
def downsample_for_plotting(data, max_points=10000):
    """
    Si hay más de 'max_points', toma muestras equiespaciadas.
    Esto reduce el peso del JSON y evita el RangeError.
    """
    n = len(data)
    if n > max_points:
        step = n // max_points
        return data[::step] # Slicing de Python: toma 1 de cada 'step'
    return data

# Configuración de página
st.set_page_config(page_title="DSP Audio Lab", layout="wide", page_icon="🎛️")

st.title("🎛️ Sistema de Procesamiento de Señales de Audio (T3)")
st.markdown("**Integrantes:** Renato Vela, Israel Méndez, Daniel Molina")

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("1. Carga de Señal")
uploaded_file = st.sidebar.file_uploader("Sube un archivo WAV", type=["wav"])

if uploaded_file is not None:
    # Cargar audio
    original_data, original_fs = load_audio(uploaded_file)
    st.sidebar.success(f"Fs Original: {original_fs} Hz")

    # --- SECCIÓN 2: CONVERSIÓN DE TASA ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. Conversión de Tasa (Resampling)")
    
    # Controles para M (Decimación) y L (Expansión)
    #[cite: 120]: Parámetros M o L establecidos interactivamente
    col1, col2 = st.sidebar.columns(2)
    L = col1.number_input("Factor L (Expansión)", min_value=1, value=1, step=1)
    M = col2.number_input("Factor M (Decimación)", min_value=1, value=1, step=1)

    # --- SECCIÓN 3: ECUALIZADOR (6 BANDAS) ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. Ecualizador")
    st.sidebar.caption("Ajuste de ganancia (dB) por banda [cite: 121]")
    
    # Sliders para las 6 bandas requeridas
    gains = {}
    gains["Sub-Bass"] = st.sidebar.slider("Sub-Bass (16-60Hz)", -20, 20, 0)
    gains["Bass"] = st.sidebar.slider("Bass (60-250Hz)", -20, 20, 0)
    gains["Low Mids"] = st.sidebar.slider("Low Mids (250-2k)", -20, 20, 0)
    gains["High Mids"] = st.sidebar.slider("High Mids (2k-4k)", -20, 20, 0)
    gains["Presence"] = st.sidebar.slider("Presence (4k-6k)", -20, 20, 0)
    gains["Brilliance"] = st.sidebar.slider("Brilliance (6k-16k)", -20, 20, 0)

    # --- PROCESAMIENTO ---
    
    # 1. Aplicar Resampling
    # Nota: Si cambiamos Fs, el ecualizador debe operar a la NUEVA Fs o a la vieja?
    # Generalmente se ecualiza la señal base, pero el ejercicio pide ver el efecto del muestreo.
    # Aplicaremos: Entrada -> Resampling -> Ecualización -> Salida
    
    resampled_data, new_fs = change_sampling_rate(original_data, original_fs, M, L)
    st.write(f"### Frecuencia de Muestreo Resultante: **{new_fs} Hz**")
    
    # 2. Aplicar Ecualización (sobre la señal re-muestreada)
    processed_data = apply_equalizer(resampled_data, new_fs, gains)

    # --- VISUALIZACIÓN ---
    
    # Pestañas para organizar la vista
    tab1, tab2 = st.tabs(["⏱️ Dominio del Tiempo", "🌊 Dominio de la Frecuencia"])

with tab1:
        st.subheader("Comparación en el Tiempo")
        
        # --- OPTIMIZACIÓN VISUAL ---
        # Graficamos un tramo representativo (ej. 1 segundo o 50k muestras) 
        # para ver la forma de onda, no toda la canción.
        limit_view = min(len(original_data), 100000) # Máximo 100k muestras para ver
        
        fig_time = go.Figure()
        # Reducimos puntos para graficar (Downsampling VISUAL)
        y_orig_plot = downsample_for_plotting(original_data[:limit_view])
        y_proc_plot = downsample_for_plotting(processed_data[:limit_view])
        
        # Eje de tiempo aproximado
        x_axis = np.linspace(0, limit_view/new_fs, len(y_proc_plot))
        
        fig_time.add_trace(go.Scatter(x=x_axis, y=y_orig_plot, name="Original", opacity=0.5))
        fig_time.add_trace(go.Scatter(x=x_axis, y=y_proc_plot, name="Procesada"))
        
        fig_time.update_layout(title="Forma de onda (Tramo inicial reducido)", xaxis_title="Tiempo (s)", yaxis_title="Amplitud")
        st.plotly_chart(fig_time, use_container_width=True)

        # --- ARREGLO DEL REPRODUCTOR DE AUDIO ---
        st.markdown("### 🎧 Escuchar Resultado")
        
        # 1. Normalizar para evitar estática o silencio (Clipping)
        # Convertimos a float32 y aseguramos rango [-1, 1]
        audio_normalized = processed_data / np.max(np.abs(processed_data))
        
        # 2. Convertir a Bytes (Archivo WAV virtual)
        # Esto engaña al navegador para que crea que cargó un archivo real
        virtual_file = io.BytesIO()
        # Convertir a formato PCM de 16 bits (estándar de audio)
        wav_data = (audio_normalized * 32767).astype(np.int16)
        write(virtual_file, new_fs, wav_data)
        
        # 3. Reproducir desde el buffer
        st.audio(virtual_file, format='audio/wav')

with tab2:
        st.subheader("Espectro de Frecuencia (FFT)")
        
        # Calcular FFT (Esto genera millones de puntos)
        freq_in, mag_in = compute_fft(original_data, original_fs)
        freq_out, mag_out = compute_fft(processed_data, new_fs)
        
        # --- SOLUCIÓN AL CRASH ---
        # Solo graficamos hasta Nyquist y reducimos la resolución visual
        # Usamos slicing [::100] o similar. 
        # Ojo: En log-log a veces perdemos detalle, pero para ver la envolvente basta.
        
        # Tomamos máximo 5000 puntos para la gráfica
        f_in_plot = downsample_for_plotting(freq_in, 5000)
        m_in_plot = downsample_for_plotting(mag_in, 5000)
        
        f_out_plot = downsample_for_plotting(freq_out, 5000)
        m_out_plot = downsample_for_plotting(mag_out, 5000)

        fig_freq = go.Figure()
        
        # Convertimos a dB y evitamos log(0)
        db_in = 20 * np.log10(m_in_plot + 1e-10)
        db_out = 20 * np.log10(m_out_plot + 1e-10)

        fig_freq.add_trace(go.Scatter(x=f_in_plot, y=db_in, name="Entrada Original"))
        fig_freq.add_trace(go.Scatter(x=f_out_plot, y=db_out, name="Salida Procesada", line=dict(color='orange')))
        
        fig_freq.update_layout(
            xaxis_title="Frecuencia (Hz)", 
            yaxis_title="Magnitud (dB)", 
            xaxis_type="log", 
            title="Comparación Espectral (Visualización Optimizada)"
        )
        st.plotly_chart(fig_freq, use_container_width=True)
else:
    st.info("👋 Sube un archivo .wav en la barra lateral para comenzar.")
