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

# Importamos el núcleo DSP con los nuevos nombres académicos
from modules.dsp_core import (
    cargar_senal_audio, 
    conversion_tasa_muestreo, 
    sistema_ecualizador, 
    calcular_espectro_magnitud
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Laboratorio de Señales y Sistemas", layout="wide", page_icon="🎛️")

st.markdown("""
    <style>
    .stAlert { display: none; } 
    .block-container { padding-top: 1rem; }
    .dsp-monitor { 
        background-color: #1e1e1e; color: #00ff00; 
        padding: 8px 12px; border-radius: 4px; 
        font-family: 'Courier New', monospace; font-size: 0.9em;
        border: 1px solid #333; margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- GESTIÓN DE ESTADO (SESSION STATE) ---
if 'senal_x' not in st.session_state:
    st.session_state.senal_x = None
    st.session_state.fs = 0
    st.session_state.nombre_archivo = ""
    st.session_state.id_sesion = str(uuid.uuid4())

# --- FUNCIONES DE INTERFAZ ---
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

# --- VISUALIZACIÓN TÉCNICA ---
def generar_reproductor_html(audio_buffer, fs, id_unico):
    b64 = base64.b64encode(audio_buffer.read()).decode()
    html = f"""
    <div class="dsp-monitor">Fs_salida: {fs} Hz | Estado: CONVERGENCIA LTI OK</div>
    <audio controls autoplay style="width:100%;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.components.v1.html(html, height=85)

def submuestreo_visual(datos, max_puntos=2500):
    """Reduce puntos solo para graficar rápido en el navegador (no afecta el audio)."""
    if len(datos) > max_puntos:
        paso = int(np.ceil(len(datos) / max_puntos))
        return datos[::paso]
    return datos

# --- INTERFAZ PRINCIPAL ---
st.title("Procesamiento de Señales en Tiempo Discreto")
st.markdown("**Asignatura:** Sistemas Lineales y Señales | **Tema:** Muestreo y Filtrado Digital")

# Panel Lateral
col_in1, col_in2 = st.sidebar.columns(2)
modo = col_in1.radio("Fuente de Señal x[n]", ["Ejemplo", "Archivo"], label_visibility="collapsed")

if modo == "Archivo":
    st.sidebar.file_uploader("Subir WAV", type=["wav"], key="uploader", on_change=callback_carga_archivo)
else:
    if os.path.exists("examples"):
        lista = [f for f in os.listdir("examples") if f.endswith('.wav')]
        if lista:
            st.sidebar.selectbox("Seleccionar", lista, key="selector_ejemplo", on_change=callback_carga_ejemplo)

if st.session_state.senal_x is None:
    st.info("👋 Cargue una señal x[n] para iniciar el análisis.")
    st.stop()

# Variables del sistema
x_n = st.session_state.senal_x
fs_entrada = st.session_state.fs

# Selección de Ventana de Análisis (Loop)
st.sidebar.markdown("---")
usar_loop = st.sidebar.checkbox("Análisis por Ventana (15s)", value=True, help="Analizar un segmento estacionario.")
if usar_loop:
    centro = len(x_n) // 2
    N_ventana = 15 * fs_entrada
    inicio = max(0, centro - (N_ventana//2))
    fin = min(len(x_n), inicio + N_ventana)
    x_trabajo = x_n[inicio:fin]
else:
    x_trabajo = x_n

# Parámetros del Sistema
c1, c2 = st.sidebar.columns(2)
L = c1.number_input("Factor Expansión (L)", 1, 8, 1, help="Interpolación ↑L")
M = c2.number_input("Factor Diezmado (M)", 1, 8, 1, help="Submuestreo ↓M")

st.sidebar.subheader("Ecualizador (Filtros LTI)")
cols_eq = st.sidebar.columns(3)
etiquetas = ["Sub", "Bass", "LoMid", "HiMid", "Pres", "Brill"]
llaves = ["Sub-Bass", "Bass", "Low Mids", "High Mids", "Presence", "Brilliance"]
ganancias = {}

for i, (lbl, key) in enumerate(zip(etiquetas, llaves)):
    with cols_eq[i%3]:
        ganancias[key] = st.slider(lbl, -15, 15, 0, key=f"g_{i}")

# --- PROCESAMIENTO (LLAMADAS AL NÚCLEO DSP) ---
with st.spinner("Calculando Convolución y Ecuaciones en Diferencias..."):
    # 1. Conversión de Tasa (Muestreo)
    x_remuestreada, fs_salida = conversion_tasa_muestreo(x_trabajo, fs_entrada, M, L)
    # 2. Ecualización (Filtrado)
    y_n = sistema_ecualizador(x_remuestreada, fs_salida, ganancias)

# --- VISUALIZACIÓN DE RESULTADOS ---
st.divider()
tipo_grafica = st.radio("Modo de Análisis:", ["Espectral y Temporal", "Secuencia Discreta (Stem)"], horizontal=True)

if tipo_grafica == "Espectral y Temporal":
    col_eje, _ = st.columns([1, 4])
    with col_eje:
        unidades = st.radio("Unidades de Frecuencia:", ["Hz (f)", "rad/s (ω)"])
        factor_escala = 2*np.pi if "rad" in unidades else 1.0

    tab1, tab2 = st.tabs(["Dominio del Tiempo (n)", "Dominio de la Frecuencia (k / ω)"])
    
    # Datos Temporales
    t_in = np.linspace(0, len(x_trabajo)/fs_entrada, len(x_trabajo))
    t_out = np.linspace(0, len(y_n)/fs_salida, len(y_n))
    
    with tab1:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=submuestreo_visual(t_in), y=submuestreo_visual(x_trabajo), 
                                   name="Entrada x[n]", line=dict(color='gray', width=1), opacity=0.5))
        fig_t.add_trace(go.Scatter(x=submuestreo_visual(t_out), y=submuestreo_visual(y_n), 
                                   name="Salida y[n]", line=dict(color='#00ff00', width=1.5)))
        fig_t.update_layout(template="plotly_dark", height=300, title="Respuesta Transitoria",
                            xaxis_title="Tiempo (s)", uirevision=st.session_state.id_sesion)
        st.plotly_chart(fig_t, use_container_width=True)

    with tab2:
        # Cálculo de Espectros (Manual)
        f_in, mag_in = calcular_espectro_magnitud(x_trabajo[:100000], fs_entrada)
        f_out, mag_out = calcular_espectro_magnitud(y_n[:100000], fs_salida)
        
        # Filtro DC y conversión a dB
        mask_in = f_in > 0.5; mask_out = f_out > 0.5
        f_in, mag_in = f_in[mask_in], 20*np.log10(mag_in[mask_in] + 1e-12)
        f_out, mag_out = f_out[mask_out], 20*np.log10(mag_out[mask_out] + 1e-12)
        
        # Ajuste de escala visual
        v_f_in = submuestreo_visual(f_in) * factor_escala
        v_f_out = submuestreo_visual(f_out) * factor_escala
        v_m_in = submuestreo_visual(mag_in)
        v_m_out = submuestreo_visual(mag_out)
        
        # Rango dinámico para líneas verticales
        y_min, y_max = -100, 20
        if len(v_m_in) > 0: 
            y_min = max(-120, np.min(v_m_in) - 10)
            y_max = min(40, np.max(v_m_in) + 10)

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=v_f_in, y=v_m_in, name="|X(jw)| Original", line=dict(color='gray'), opacity=0.6))
        fig_f.add_trace(go.Scatter(x=v_f_out, y=v_m_out, name="|Y(jw)| Procesada", fill='tozeroy', line=dict(color='cyan')))
        
        # Bandas de Ecualización (Líneas Naranja)
        bandas_hz = [60, 250, 2000, 4000, 6000]
        color_linea = "#FF5500"
        
        for b_hz in bandas_hz:
            pos = b_hz * factor_escala
            etiqueta = f"{b_hz} Hz" if "Hz" in unidades else f"{int(b_hz)} rad/s"
            fig_f.add_trace(go.Scatter(
                x=[pos, pos], y=[y_min, y_max], mode="lines",
                name=f"Corte {etiqueta}",
                line=dict(color=color_linea, width=1.5, dash="dash"),
                hoverinfo="name"
            ))

        fig_f.update_layout(template="plotly_dark", height=350, title="Espectro de Magnitud",
                            xaxis=dict(type="log", title=f"Frecuencia ({unidades.split()[0]})"),
                            yaxis_title="Magnitud (dB)", legend=dict(x=1, y=1),
                            uirevision=st.session_state.id_sesion)
        st.plotly_chart(fig_f, use_container_width=True)

else:
    # MODO TEÓRICO (STEM PLOTS)
    st.markdown("#### 🔬 Representación de Secuencias Discretas (Zoom)")
    st.caption("Visualización de muestras individuales como impulsos discretos (Oppenheim Cap. 1)")
    
    muestras = 40
    c = len(x_trabajo) // 2
    
    x_stem = x_trabajo[c : c+muestras]
    ratio = fs_salida / fs_entrada
    c_out = int(c * ratio)
    m_out = int(muestras * ratio)
    y_stem = y_n[c_out : c_out+m_out]
    
    # Normalización visual
    x_stem /= (np.max(np.abs(x_stem)) + 1e-9)
    y_stem /= (np.max(np.abs(y_stem)) + 1e-9)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.stem(range(len(x_stem)), x_stem, linefmt='k-', markerfmt='ko', basefmt='k-')
    ax1.set_title(r"Entrada $x[n]$", fontsize=12)
    ax1.set_ylabel("Amplitud Normalizada")
    ax1.grid(alpha=0.3)
    
    eje_salida = np.linspace(0, len(x_stem), len(y_stem))
    ax2.stem(eje_salida, y_stem, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax2.set_title(r"Salida $y[n]$ (Interpolada/Diezmada)", fontsize=12)
    ax2.set_xlabel("n (muestras relativas)")
    ax2.grid(alpha=0.3)
    
    st.pyplot(fig)

# --- EXPORTACIÓN ---
st.divider()
c_out1, c_out2 = st.columns([3, 1])
with c_out1:
    # Preparar audio final
    y_final = np.nan_to_num(y_n)
    peak = np.max(np.abs(y_final))
    if peak > 0: y_final /= peak
    
    buffer = io.BytesIO()
    write(buffer, fs_salida, (y_final * 32767).astype(np.int16))
    generar_reproductor_html(buffer, fs_salida, st.session_state.id_sesion)

with c_out2:
    st.download_button("💾 Descargar WAV", buffer, "senal_procesada.wav", "audio/wav")

# Limpieza GC
del x_remuestreada, y_n, y_final, buffer
gc.collect()
