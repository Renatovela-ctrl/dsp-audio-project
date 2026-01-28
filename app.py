import streamlit as st
import numpy as np
import plotly.graph_objs as go
import io
import os
import gc # <--- NUEVO: Garbage Collector para liberar RAM
from scipy.io.wavfile import write
from modules.dsp_core import load_audio, change_sampling_rate, apply_equalizer, compute_fft

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="DSP Audio Lab", layout="wide", page_icon="🎛️")

# Estilos CSS para ocultar elementos molestos y warnings
st.markdown("""
    <style>
    .stAlert { display: none; }
    .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- CACHÉ INTELIGENTE ---
@st.cache_data(show_spinner=False)
def cached_resampling(data, fs, m, l):
    # Esta función consume mucha RAM, el caché evita recalcular si no cambias M o L
    return change_sampling_rate(data, fs, m, l)

def downsample_visuals(data, max_points=3000): # <--- OPTIMIZACIÓN: Bajamos a 3000 puntos
    """
    Reduce drásticamente los puntos para graficar.
    3000 puntos son suficientes para el ojo humano en una pantalla 4K.
    """
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

# --- INTERFAZ ---
st.title("🎛️ DSP: Conversor de Tasa y Ecualizador")
st.markdown("**Equipo:** Renato Vela, Israel Méndez, Daniel Molina")

# --- SIDEBAR ---
st.sidebar.header("1. Entrada de Audio")
input_mode = st.sidebar.radio("Fuente:", ["📂 Subir Archivo", "🎵 Usar Ejemplo"])

input_data = None 

if input_mode == "📂 Subir Archivo":
    uploaded_file = st.sidebar.file_uploader("Sube un archivo WAV", type=["wav"])
    if uploaded_file is not None:
        input_data = uploaded_file
else:
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        files = [f for f in os.listdir(examples_dir) if f.endswith('.wav')]
        if files:
            selected_file = st.sidebar.selectbox("Elige un audio:", files)
            input_data = os.path.join(examples_dir, selected_file)

# --- LÓGICA PRINCIPAL CON GESTIÓN DE MEMORIA ---
if input_data is not None:
    # Carga
    original_data, original_fs = load_audio(input_data)
    
    # Mostrar info básica sin saturar
    st.sidebar.success(f"Fs: {original_fs} Hz | Duración: {len(original_data)/original_fs:.2f}s")

    st.sidebar.markdown("---")
    st.sidebar.header("2. Resampling")
    col1, col2 = st.sidebar.columns(2)
    # Usamos keys únicas para evitar conflictos de estado
    L = col1.number_input("Expansión (L)", 1, 10, 1, key="L_val")
    M = col2.number_input("Decimación (M)", 1, 10, 1, key="M_val")

    st.sidebar.markdown("---")
    st.sidebar.header("3. Ecualizador (dB)")
    
    # Formulario para evitar recálculos en cada milímetro de movimiento (OPCIONAL)
    # Por ahora lo dejamos directo, pero si sigue lento, podemos meter esto en un st.form
    gains = {
        "Sub-Bass": st.sidebar.slider("16-60Hz", -12, 12, 0),
        "Bass": st.sidebar.slider("60-250Hz", -12, 12, 0),
        "Low Mids": st.sidebar.slider("250-2k", -12, 12, 0),
        "High Mids": st.sidebar.slider("2k-4k", -12, 12, 0),
        "Presence": st.sidebar.slider("4k-6k", -12, 12, 0),
        "Brilliance": st.sidebar.slider("6k-16k", -12, 12, 0),
    }

    # --- PROCESAMIENTO ---
    # 1. Resampling
    resampled_data, new_fs = cached_resampling(original_data, original_fs, M, L)
    
    # 2. Ecualización
    processed_data = apply_equalizer(resampled_data, new_fs, gains)

    # --- VISUALIZACIÓN LIGERA ---
    st.divider()
    
    # Layout en columnas para aprovechar espacio
    col_graph, col_controls = st.columns([3, 1])
    
    with col_graph:
        tab_t, tab_f = st.tabs(["⏱️ Tiempo", "🌊 Frecuencia"])

        with tab_t:
            # Límite visual: Solo graficar los primeros 5 segundos o 100k muestras
            # Esto evita que el navegador explote con canciones largas
            limit = min(len(original_data), 200000)
            
            fig = go.Figure()
            
            # Datos reducidos para visualización
            vis_orig = downsample_visuals(original_data[:limit])
            vis_proc = downsample_visuals(processed_data[:limit])
            
            # Ejes de tiempo aproximados
            t_orig = np.linspace(0, limit/original_fs, num=len(vis_orig))
            t_proc = np.linspace(0, limit/original_fs, num=len(vis_proc))
            
            fig.add_trace(go.Scatter(y=vis_orig, x=t_orig, name="Original", opacity=0.5))
            fig.add_trace(go.Scatter(y=vis_proc, x=t_proc, name="Procesada"))
            
            fig.update_layout(
                title="Comparativa Temporal (Zoom primeros segs)", 
                margin=dict(l=0, r=0, t=30, b=0),
                height=350,
                legend=dict(orientation="h", y=1, x=0)
            )
            # CORRECCIÓN DE WARNING: Usamos el parámetro moderno si está disponible
            try:
                st.plotly_chart(fig, use_container_width=True)
            except TypeError:
                st.plotly_chart(fig) 

        with tab_f:
            # FFT
            f_in, mag_in = compute_fft(original_data, original_fs)
            f_out, mag_out = compute_fft(processed_data, new_fs)
            
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(x=downsample_visuals(f_in), y=downsample_visuals(20*np.log10(mag_in+1e-9)), name="Entrada"))
            fig_fft.add_trace(go.Scatter(x=downsample_visuals(f_out), y=downsample_visuals(20*np.log10(mag_out+1e-9)), name="Salida"))
            
            fig_fft.update_layout(
                xaxis_type="log", 
                title="Espectro (dB)", 
                margin=dict(l=0, r=0, t=30, b=0), 
                height=350,
                legend=dict(orientation="h", y=1, x=0)
            )
            try:
                st.plotly_chart(fig_fft, use_container_width=True)
            except:
                st.plotly_chart(fig_fft)

    with col_controls:
        st.write(f"### 🎧 Salida")
        st.caption(f"Frecuencia: **{new_fs} Hz**")
        
        # PREPARACIÓN DE AUDIO (RAM EFFICIENT)
        # 1. Limpieza
        audio_final = np.nan_to_num(processed_data)
        
        # 2. Normalizar (Solo si es necesario)
        max_amp = np.max(np.abs(audio_final))
        if max_amp > 0: audio_final /= max_amp
        
        # 3. Clip
        audio_final = np.clip(audio_final, -1.0, 1.0)
        
        # 4. Write
        virtual_wav = io.BytesIO()
        write(virtual_wav, new_fs, (audio_final * 32760).astype(np.int16))
        virtual_wav.seek(0)
        
        st.audio(virtual_wav, format='audio/wav')
        
        # Botón de descarga explícito
        st.download_button(
            label="⬇️ Descargar WAV",
            data=virtual_wav,
            file_name="audio_procesado.wav",
            mime="audio/wav"
        )

    # --- LIMPIEZA DE MEMORIA MANUAL ---
    # Esto ayuda a que el servidor no se quede con basura de la ejecución anterior
    del original_data
    del resampled_data
    del processed_data
    gc.collect()

else:
    st.info("👈 Selecciona un archivo para comenzar.")
