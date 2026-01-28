import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="DSP Audio Lab", layout="wide", page_icon="🎛️")

# Estilos CSS para ocultar warnings molestos
st.markdown("""
    <style>
    .stAlert { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- CACHÉ PARA PROCESOS PESADOS ---
# Esto evita recalcular el resampling si solo cambias de pestaña
@st.cache_data
def cached_resampling(data, fs, m, l):
    return change_sampling_rate(data, fs, m, l)

def downsample_visuals(data, max_points=5000):
    """Reduce puntos para gráficas rápidas"""
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

# --- INTERFAZ ---
st.title("🎛️ DSP: Conversor de Tasa y Ecualizador")
st.markdown("**Equipo:** Renato Vela, Israel Méndez, Daniel Molina")

# --- SIDEBAR ---
st.sidebar.header("1. Entrada")
uploaded_file = st.sidebar.file_uploader("Archivo WAV", type=["wav"])

if uploaded_file:
    # Carga Robustecida
    original_data, original_fs = load_audio(uploaded_file)
    st.sidebar.success(f"Fs Original: {original_fs} Hz")

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
    # 1. Resampling (Con Caché)
    resampled_data, new_fs = cached_resampling(original_data, original_fs, M, L)
    
    st.metric("Nueva Frecuencia de Muestreo", f"{new_fs} Hz")

    # 2. Ecualización (En tiempo real)
    processed_data = apply_equalizer(resampled_data, new_fs, gains)

    # --- SALIDAS VISUALES ---
    tab_t, tab_f = st.tabs(["⏱️ Tiempo", "🌊 Frecuencia"])

    with tab_t:
        # Gráfica optimizada
        limit = min(len(original_data), 100000) # Ver solo primeros segundos
        
        fig = go.Figure()
        
        # Ejes de tiempo correctos
        t_orig = np.linspace(0, limit/original_fs, num=len(downsample_visuals(original_data[:limit])))
        t_proc = np.linspace(0, limit/original_fs, num=len(downsample_visuals(processed_data[:limit])))
        
        fig.add_trace(go.Scatter(y=downsample_visuals(original_data[:limit]), x=t_orig, name="Original", opacity=0.5))
        fig.add_trace(go.Scatter(y=downsample_visuals(processed_data[:limit]), x=t_proc, name="Procesada"))
        fig.update_layout(title="Comparativa Temporal", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # --- REPRODUCTOR DE AUDIO (FIX DEFINITIVO) ---
        st.markdown("### 🎧 Resultado")
        
        # 1. Limpieza matemática (Quitar NaNs/Infs)
        audio_final = np.nan_to_num(processed_data)
        
        # 2. Normalización segura (Evitar volumen bajo o saturado)
        max_amp = np.max(np.abs(audio_final))
        if max_amp > 0:
            audio_final = audio_final / max_amp
            
        # 3. Clipping estricto (Evita el error 'overflow encountered in multiply')
        audio_final = np.clip(audio_final, -1.0, 1.0)
        
        # 4. Conversión y Escritura
        virtual_wav = io.BytesIO()
        # Multiplicamos por 32760 (un poco menos de 32767 por seguridad)
        int_data = (audio_final * 32760).astype(np.int16)
        write(virtual_wav, new_fs, int_data)
        
        # 5. EL REBOBINADO CRÍTICO
        virtual_wav.seek(0)
        
        st.audio(virtual_wav, format='audio/wav')

    with tab_f:
        # FFT
        f_in, mag_in = compute_fft(original_data, original_fs)
        f_out, mag_out = compute_fft(processed_data, new_fs)
        
        fig_fft = go.Figure()
        # Usamos decibelios logarítmicos
        fig_fft.add_trace(go.Scatter(x=downsample_visuals(f_in), y=downsample_visuals(20*np.log10(mag_in+1e-9)), name="Entrada"))
        fig_fft.add_trace(go.Scatter(x=downsample_visuals(f_out), y=downsample_visuals(20*np.log10(mag_out+1e-9)), name="Salida"))
        
        fig_fft.update_layout(xaxis_type="log", title="Espectro de Magnitud (dB)", yaxis_title="dB", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_fft, use_container_width=True)

else:
    st.info("Sube un archivo WAV para comenzar el análisis.")
