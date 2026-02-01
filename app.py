import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import io
import os
import gc
import base64
import uuid
from scipy.io.wavfile import write

# Importación de funciones académicas desde el núcleo
from modules.dsp_core import (
    cargar_senal_audio, 
    conversion_tasa_muestreo, 
    sistema_ecualizador, 
    calcular_espectro_magnitud
)

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Laboratorio DSP - UCuenca", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; } 
    .block-container { padding-top: 1.5rem; }
    .dsp-monitor { 
        background-color: #1e1e1e; color: #00ff00; 
        padding: 10px 15px; border-radius: 5px; 
        font-family: 'Courier New', monospace; font-size: 0.9em;
        border: 1px solid #333; margin-bottom: 15px;
    }
    .stSlider { margin-bottom: -15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO ---
if 'senal_x' not in st.session_state:
    st.session_state.senal_x = None
    st.session_state.fs = 0
    st.session_state.nombre_archivo = ""
    st.session_state.id_sesion = str(uuid.uuid4())

# --- 3. CALLBACKS DE CARGA ---
def actualizar_senal(x, fs, nombre):
    st.session_state.senal_x = x
    st.session_state.fs = fs
    st.session_state.nombre_archivo = nombre
    st.session_state.id_sesion = str(uuid.uuid4())

def callback_carga_archivo():
    if st.session_state.uploader:
        x, fs = cargar_senal_audio(st.session_state.uploader)
        actualizar_senal(x, fs, st.session_state.uploader.name)

def callback_carga_ejemplo():
    ruta = os.path.join("examples", st.session_state.selector_ejemplo)
    if os.path.exists(ruta):
        x, fs = cargar_senal_audio(ruta)
        actualizar_senal(x, fs, st.session_state.selector_ejemplo)

# --- 4. UTILIDADES ---
def generar_reproductor_html(audio_buffer, fs, id_unico):
    b64 = base64.b64encode(audio_buffer.read()).decode()
    html = f"""
    <div class="dsp-monitor">Fs_salida: {fs} Hz</div>
    <audio controls autoplay style="width:100%;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.components.v1.html(html, height=85)

def submuestreo_visual(datos, max_puntos=2500):
    if len(datos) > max_puntos:
        paso = int(np.ceil(len(datos) / max_puntos))
        return datos[::paso]
    return datos

# --- 5. INTERFAZ PRINCIPAL ---
st.title("🎛️ Procesamiento de Señales de Audio en Tiempo Discreto")
st.markdown("""
**Autores:** Israel Méndez, Daniel Molina, Renato Vela  
**Asignatura:** Sistemas Lineales y Señales - Facultad de Ingeniería (UCUENCA)
""")

# PANEL LATERAL
st.sidebar.header("Configuración de Entrada")
col_modo1, col_modo2 = st.sidebar.columns(2)
modo = col_modo1.radio("Fuente x[n]", ["Ejemplo", "Archivo"], label_visibility="collapsed")

if modo == "Archivo":
    st.sidebar.file_uploader("Subir WAV", type=["wav"], key="uploader", on_change=callback_carga_archivo)
else:
    if os.path.exists("examples"):
        lista = [f for f in os.listdir("examples") if f.endswith('.wav')]
        if lista:
            st.sidebar.selectbox("Seleccionar Audio", lista, key="selector_ejemplo", on_change=callback_carga_ejemplo)

if st.session_state.senal_x is None:
    st.info("👋 Bienvenid@. Cargue una señal para iniciar el procesamiento.")
    st.stop()

x_n = st.session_state.senal_x
fs_entrada = st.session_state.fs

st.sidebar.markdown("---")
usar_loop = st.sidebar.checkbox("Análisis por Ventana (15s)", value=False, help="Analizar solo un segmento central.")
if usar_loop:
    centro = len(x_n) // 2
    N_ventana = 15 * fs_entrada
    inicio = max(0, centro - (N_ventana//2))
    fin = min(len(x_n), inicio + N_ventana)
    x_trabajo = x_n[inicio:fin]
else:
    x_trabajo = x_n

st.sidebar.subheader("Conversor de Tasa")
c1, c2 = st.sidebar.columns(2)
L = c1.number_input("Expansión (L)", 1, 8, 1)
M = c2.number_input("Diezmado (M)", 1, 8, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Ecualizador (Ganancia dB)")
llaves = ["Sub-Bass", "Bass", "Low Mids", "High Mids", "Presence", "Brilliance"]
rangos = ["16-60 Hz", "60-250 Hz", "250-2k Hz", "2k-4k Hz", "4k-6k Hz", "6k-16k Hz"]
ganancias = {}

for i, (key, rango) in enumerate(zip(llaves, rangos)):
    ganancias[key] = st.sidebar.slider(f"{key} ({rango})", -15, 15, 0, key=f"g_{i}")

# --- PROCESAMIENTO ---
with st.spinner("Procesando señal..."):
    x_remuestreada, fs_salida = conversion_tasa_muestreo(x_trabajo, fs_entrada, M, L)
    y_n = sistema_ecualizador(x_remuestreada, fs_salida, ganancias)

# --- VISUALIZACIÓN ---
st.divider()
tipo_grafica = st.radio("Modo de Análisis:", ["Espectral y Temporal", "Secuencia Discreta (Stem)"], horizontal=True)

if tipo_grafica == "Espectral y Temporal":
    col_eje, _ = st.columns([1, 4])
    with col_eje:
        unidades = st.radio("Unidades:", ["Hz (f)", "rad/s (ω)"], horizontal=True)
        factor_escala = 2*np.pi if "rad" in unidades else 1.0

    tab1, tab2 = st.tabs(["Dominio del Tiempo", "Dominio de la Frecuencia"])
    
    with tab1:
        t_in = np.linspace(0, len(x_trabajo)/fs_entrada, len(x_trabajo))
        t_out = np.linspace(0, len(y_n)/fs_salida, len(y_n))
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=submuestreo_visual(t_in), y=submuestreo_visual(x_trabajo), 
                                   name="x[n]", line=dict(color='gray', width=1), opacity=0.5))
        fig_t.add_trace(go.Scatter(x=submuestreo_visual(t_out), y=submuestreo_visual(y_n), 
                                   name="y[n]", line=dict(color='#00ff00', width=1.5)))
        fig_t.update_layout(template="plotly_dark", height=300, title="Señales en el Tiempo",
                            xaxis_title="Tiempo (s)", uirevision=st.session_state.id_sesion)
        st.plotly_chart(fig_t, use_container_width=True)

    with tab2:
        f_in, mag_in = calcular_espectro_magnitud(x_trabajo[:100000], fs_entrada)
        f_out, mag_out = calcular_espectro_magnitud(y_n[:100000], fs_salida)
        
        mask_in = f_in > 0.5; mask_out = f_out > 0.5
        db_in = 20*np.log10(mag_in[mask_in] + 1e-12)
        db_out = 20*np.log10(mag_out[mask_out] + 1e-12)
        
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=submuestreo_visual(f_in[mask_in]) * factor_escala, y=submuestreo_visual(db_in), 
                                   name="|X(jw)|", line=dict(color='gray'), opacity=0.6))
        fig_f.add_trace(go.Scatter(x=submuestreo_visual(f_out[mask_out]) * factor_escala, y=submuestreo_visual(db_out), 
                                   name="|Y(jw)|", fill='tozeroy', line=dict(color='cyan')))
        
        bandas_hz = [60, 250, 2000, 4000, 6000]
        for b_hz in bandas_hz:
            pos = b_hz * factor_escala
            fig_f.add_vline(x=pos, line_dash="dash", line_color="#FF5500", opacity=0.7)

        fig_f.update_layout(template="plotly_dark", height=350, title="Análisis Espectral",
                            xaxis=dict(type="log", title=f"Frecuencia ({unidades.split()[0]})"),
                            yaxis_title="Magnitud (dB)", uirevision=st.session_state.id_sesion)
        st.plotly_chart(fig_f, use_container_width=True)

else:
    # MODO TEÓRICO (STEM + ANGULAR)
    st.markdown("#### 🔬 Representación de Secuencias y Espectro Angular")
    
    # --- 1. STEM PLOT (TIEMPO) ---
    muestras = 40
    # Centro de la señal (donde hay audio garantizado)
    c = len(x_trabajo) // 2
    
    # Datos centrados
    x_s = x_trabajo[c:c+muestras]
    
    # Equivalente en salida
    ratio = fs_salida / fs_entrada
    c_out = int(c * ratio)
    m_out = int(muestras * ratio)
    y_s = y_n[c_out : c_out + m_out]

    fig_stem, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
    
    norm_x = np.max(np.abs(x_s)) if np.max(np.abs(x_s)) > 0 else 1
    norm_y = np.max(np.abs(y_s)) if np.max(np.abs(y_s)) > 0 else 1

    ax1.stem(range(len(x_s)), x_s/norm_x, linefmt='k-', markerfmt='ko', basefmt='k-')
    ax1.set_title(r"Entrada $x[n]$ (Normalizada)", fontsize=10)
    ax1.grid(alpha=0.3)

    eje_salida = np.linspace(0, len(x_s), len(y_s))
    ax2.stem(eje_salida, y_s/norm_y, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax2.set_title(r"Salida $y[n]$ (Normalizada)", fontsize=10)
    ax2.grid(alpha=0.3)
    
    st.pyplot(fig_stem)

    # --- 2. ESPECTRO ANGULAR (CORRECCIÓN: DATOS CENTRADOS) ---
    st.markdown("#### 📐 Espectro Angular ($-\pi$ a $\pi$)")
    
    N_fft = 1024
    
    # --- CORRECCIÓN CLAVE AQUÍ ---
    # Antes tomábamos [:N_fft] (el inicio, que puede ser silencio).
    # Ahora tomamos [c - 512 : c + 512] (el centro, donde hay música).
    start_idx = max(0, c - N_fft // 2)
    end_idx = min(len(x_trabajo), start_idx + N_fft)
    
    # Segmento Entrada
    seg_in = x_trabajo[start_idx : end_idx]
    # Relleno si faltan muestras
    if len(seg_in) < N_fft: seg_in = np.pad(seg_in, (0, N_fft - len(seg_in)))
    
    # Segmento Salida (ajustado por ratio)
    start_out = int(start_idx * ratio)
    len_out = int(N_fft * ratio) # Aproximado para visualizar
    # Limitamos para no salir del array
    if start_out + len_out > len(y_n):
        start_out = max(0, len(y_n) - len_out)
    
    seg_out = y_n[start_out : start_out + len_out]
    # Relleno para salida
    target_len_out = len(seg_out) # Usamos longitud real obtenida
    
    W_in = np.fft.fftshift(np.fft.fft(seg_in))
    W_out = np.fft.fftshift(np.fft.fft(seg_out)) # FFT de longitud variable según resampling
    
    w_axis_in = np.linspace(-np.pi, np.pi, len(W_in))
    w_axis_out = np.linspace(-np.pi, np.pi, len(W_out))
    
    fig_w, ax3 = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax3.plot(w_axis_in, 20*np.log10(np.abs(W_in)+1e-9), 'k--', alpha=0.5, label='Entrada')
    ax3.plot(w_axis_out, 20*np.log10(np.abs(W_out)+1e-9), 'r-', label='Salida')
    
    ax3.set_xlim(-np.pi, np.pi)
    ax3.set_xlabel(r"Frecuencia $\omega$ (rad)")
    ax3.set_xticks([-np.pi, 0, np.pi])
    ax3.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax3.set_ylabel("Magnitud (dB)")
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3)
    
    st.pyplot(fig_w)

# --- REPRODUCCIÓN Y DESCARGA ---
st.divider()
c_sal1, c_sal2 = st.columns([3, 1])
with c_sal1:
    y_final = np.nan_to_num(y_n)
    peak = np.max(np.abs(y_final))
    if peak > 0: y_final /= peak
    
    buffer = io.BytesIO()
    write(buffer, fs_salida, (y_final * 32767).astype(np.int16))
    generar_reproductor_html(buffer, fs_salida, st.session_state.id_sesion)

with c_sal2:
    st.download_button("💾 Descargar WAV", buffer, "salida_dsp.wav", "audio/wav")

gc.collect()
