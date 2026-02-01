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

# Importación de funciones manuales con nombres en español
from modules.dsp_core import cargar_audio, cambiar_tasa_muestreo, aplicar_ecualizador, calcular_fft

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Laboratorio DSP", layout="wide", page_icon="🎛️")

# Estilos CSS personalizados
st.markdown("""
    <style>
    .stAlert { display: none; } 
    .block-container { padding-top: 1rem; }
    .dsp-monitor { 
        background-color: #222; color: #0f0; 
        padding: 8px 12px; border-radius: 4px; 
        font-family: 'Consolas', monospace; font-size: 0.9em;
        border: 1px solid #444; margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ESTADO DE LA APLICACIÓN ---
if 'datos_audio' not in st.session_state:
    st.session_state.datos_audio = None
    st.session_state.fs = 0
    st.session_state.nombre_archivo = ""
    st.session_state.id_archivo = str(uuid.uuid4())

# --- 3. FUNCIONES DE CARGA ---
def nuevo_archivo_cargado(datos, fs, nombre):
    st.session_state.datos_audio = datos
    st.session_state.fs = fs
    st.session_state.nombre_archivo = nombre
    # Generar nuevo ID para resetear vistas
    st.session_state.id_archivo = str(uuid.uuid4())

def cargar_desde_subida():
    if st.session_state.uploader:
        d, fs = cargar_audio(st.session_state.uploader)
        nuevo_archivo_cargado(d, fs, st.session_state.uploader.name)

def cargar_desde_ejemplo():
    ruta = os.path.join("examples", st.session_state.selector_ejemplo)
    if os.path.exists(ruta):
        d, fs = cargar_audio(ruta)
        nuevo_archivo_cargado(d, fs, st.session_state.selector_ejemplo)

# --- 4. UTILIDADES GRÁFICAS ---
def normalizar_visual(datos):
    maximo = np.max(np.abs(datos))
    if maximo > 0: return datos / maximo
    return datos

def diezmar_para_grafica(datos, max_puntos=2000):
    if len(datos) > max_puntos:
        paso = int(np.ceil(len(datos) / max_puntos))
        return datos[::paso]
    return datos

def renderizar_reproductor(audio_bytes, fs, id_unico):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    id_html = f"audio_{id_unico}"
    clave_almacenamiento = f"tiempo_{id_unico}"
    
    # HTML/JS para reproductor persistente
    codigo_html = f"""
    <div class="dsp-monitor">SALIDA: {fs} Hz | ESTADO: LISTO</div>
    <audio id="{id_html}" controls autoplay style="width:100%;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    <script>
        (function() {{
            var a = document.getElementById('{id_html}');
            var k = '{clave_almacenamiento}';
            a.onloadedmetadata = function() {{
                var s = sessionStorage.getItem(k);
                if(s && s!=="null") {{
                    var t = parseFloat(s);
                    if(t < a.duration) a.currentTime = t;
                }}
                a.play().catch(e=>console.log("Esperando interacción"));
            }};
            a.ontimeupdate = function() {{ sessionStorage.setItem(k, a.currentTime); }};
        }})();
    </script>
    """
    st.components.v1.html(codigo_html, height=85)

# --- 5. INTERFAZ DE USUARIO ---
st.title("🎛️ Conversor de Frecuencia de Muestreo y Ecualizador - Israel Méndez, Daniel Molina, Renato Vela")

# Panel lateral: Entrada
col_ent1, col_ent2 = st.sidebar.columns(2)
modo_entrada = col_ent1.radio("Fuente", ["Ejemplo", "Subir"], label_visibility="collapsed")

if modo_entrada == "Subir":
    st.sidebar.file_uploader("Archivo WAV", type=["wav"], key="uploader", on_change=cargar_desde_subida)
else:
    if os.path.exists("examples"):
        archivos = [f for f in os.listdir("examples") if f.endswith('.wav')]
        if archivos:
            st.sidebar.selectbox("Seleccionar", archivos, key="selector_ejemplo", on_change=cargar_desde_ejemplo)

if st.session_state.datos_audio is None:
    st.info("⚠️ Por favor carga una señal de audio para comenzar.")
    st.stop()

# Variables principales
datos_crudos = st.session_state.datos_audio
fs_entrada = st.session_state.fs

# Configuración de Bucle (Loop)
st.sidebar.markdown("---")
usar_bucle = st.sidebar.checkbox("Modo Bucle (15s)", value=True, help="Recomendado para agilizar el procesamiento manual.")
if usar_bucle:
    centro = len(datos_crudos) // 2
    ventana = 15 * fs_entrada
    inicio = max(0, centro - (ventana//2))
    fin = min(len(datos_crudos), inicio + ventana)
    datos_trabajo = datos_crudos[inicio:fin]
else:
    datos_trabajo = datos_crudos

# Controles de Muestreo
c1, c2 = st.sidebar.columns(2)
L = c1.number_input("Expansión (L)", 1, 8, 1)
M = c2.number_input("Decimación (M)", 1, 8, 1)

# Controles de Ecualización
st.sidebar.subheader("Banco de Filtros")
nombres_bandas = ["Sub", "Bass", "LoMid", "HiMid", "Pres", "Brill"]
llaves_bandas = ["Sub-Bass", "Bass", "Low Mids", "High Mids", "Presence", "Brilliance"]
ganancias = {}
columnas_eq = st.sidebar.columns(3)

for i, (etiqueta, llave) in enumerate(zip(nombres_bandas, llaves_bandas)):
    with columnas_eq[i%3]:
        ganancias[llave] = st.slider(etiqueta, -15, 15, 0, key=f"eq_{i}")

# --- PROCESAMIENTO DSP (MANUAL) ---
with st.spinner("Calculando filtros manualmente..."):
    datos_remuestreados, fs_salida = cambiar_tasa_muestreo(datos_trabajo, fs_entrada, M, L)
    datos_procesados = aplicar_ecualizador(datos_remuestreados, fs_salida, ganancias)

# --- VISUALIZACIÓN ---
st.divider()
modo_visual = st.radio("Modo Visual:", ["🛠️ Análisis Completo", "📖 Teórico (50 muestras)"], horizontal=True)

if modo_visual == "🛠️ Análisis Completo":
    col_opciones, _ = st.columns([1, 4])
    with col_opciones:
        unidad_frec = st.radio("Eje X:", ["Hz", "rad/s"])
        mult_x = 2*np.pi if unidad_frec == "rad/s" else 1.0

    pestana1, pestana2 = st.tabs(["Dominio del Tiempo", "Dominio de la Frecuencia"])
    
    # Preparación de datos temporales
    vis_entrada = diezmar_para_grafica(normalizar_visual(datos_trabajo))
    vis_salida = diezmar_para_grafica(normalizar_visual(datos_procesados))
    eje_t_in = np.linspace(0, len(vis_entrada)/fs_entrada, len(vis_entrada))
    eje_t_out = np.linspace(0, len(vis_salida)/fs_salida, len(vis_salida))

    with pestana1:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=eje_t_in, y=vis_entrada, name="Entrada", line=dict(color='gray', width=1), opacity=0.5))
        fig_t.add_trace(go.Scatter(x=eje_t_out, y=vis_salida, name="Salida", line=dict(color='#0f0', width=1.5)))
        fig_t.update_layout(
            template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10),
            title="Comparativa Temporal", uirevision=st.session_state.id_archivo
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with pestana2:
        # Cálculo de FFT Manual
        limite = min(len(datos_trabajo), 100000)
        f_in, mag_in = calcular_fft(datos_trabajo[:limite], fs_entrada)
        f_out, mag_out = calcular_fft(datos_procesados[:limite], fs_salida)
        
        # Filtrado de componente DC (0 Hz) para evitar errores logarítmicos
        mascara_in = f_in > 0.5 
        mascara_out = f_out > 0.5
        f_in, mag_in = f_in[mascara_in], mag_in[mascara_in]
        f_out, mag_out = f_out[mascara_out], mag_out[mascara_out]
        
        # Conversión a Decibelios
        db_in = 20*np.log10(mag_in + 1e-9)
        db_out = 20*np.log10(mag_out + 1e-9)
        
        # Reducción de puntos para gráfica web
        v_f_in = diezmar_para_grafica(f_in) * mult_x
        v_db_in = diezmar_para_grafica(db_in)
        v_f_out = diezmar_para_grafica(f_out) * mult_x
        v_db_out = diezmar_para_grafica(db_out)
        
        # Calcular límites Y para las líneas verticales
        if len(v_db_in) > 0 and len(v_db_out) > 0:
            y_min = min(np.min(v_db_in), np.min(v_db_out)) - 5
            y_max = max(np.max(v_db_in), np.max(v_db_out)) + 5
        else:
            y_min, y_max = -100, 10

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=v_f_in, y=v_db_in, name="Original", line=dict(color='gray'), opacity=0.6))
        fig_f.add_trace(go.Scatter(x=v_f_out, y=v_db_out, name="Procesada", fill='tozeroy', line=dict(color='cyan')))
        
        # Líneas verticales indicando bandas
        limites_bandas = [60, 250, 2000, 4000, 6000]
        color_linea = "#FF5500" # Naranja brillante
        
        for borde in limites_bandas:
            pos = borde * mult_x
            etiqueta = f"{borde} Hz" if unidad_frec == "Hz" else f"{int(borde)} rad/s"
            
            # Dibujar línea como trazo para tener leyenda y hover
            fig_f.add_trace(go.Scatter(
                x=[pos, pos], 
                y=[y_min, y_max], 
                mode="lines",
                name=etiqueta,
                line=dict(color=color_linea, width=1.5, dash="dash"),
                hoverinfo="name" 
            ))

        fig_f.update_layout(
            template="plotly_dark", height=350, margin=dict(l=10, r=10, t=30, b=10),
            title=f"Espectro de Magnitud ({unidad_frec})", 
            xaxis=dict(type="log", title=f"Frecuencia ({unidad_frec})"),
            yaxis=dict(title="Magnitud (dB)"), 
            legend=dict(x=1, y=1), 
            uirevision=st.session_state.id_archivo
        )
        st.plotly_chart(fig_f, use_container_width=True)

else:
    # MODO TEÓRICO (STEM PLOTS)
    st.markdown("#### 🔬 Análisis Discreto (Zoom 50 muestras)")
    n_muestras = 40
    centro = len(datos_trabajo) // 2
    
    recorte_in = normalizar_visual(datos_trabajo[centro : centro+n_muestras])
    
    ratio = fs_salida / fs_entrada
    centro_out = int(centro * ratio)
    n_out = int(n_muestras * ratio)
    recorte_out = normalizar_visual(datos_procesados[centro_out : centro_out+n_out])

    fig_stem, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
    
    ax1.stem(range(len(recorte_in)), recorte_in, linefmt='k-', markerfmt='ko', basefmt='k-')
    ax1.set_title(r"Entrada $x[n]$", fontsize=10)
    ax1.grid(alpha=0.3)
    
    idx_out = np.linspace(0, len(recorte_in), len(recorte_out))
    ax2.stem(idx_out, recorte_out, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax2.set_title(r"Salida $y[n]$", fontsize=10)
    ax2.grid(alpha=0.3)
    
    st.pyplot(fig_stem)

    st.markdown("#### 📐 Espectro Angular ($-\pi$ a $\pi$)")
    N_fft = 1024
    # Usamos numpy fft solo para esta visualización estática rápida, 
    # el procesamiento real ya se hizo con la función manual.
    W_in = np.fft.fftshift(np.fft.fft(datos_trabajo[:N_fft]))
    W_out = np.fft.fftshift(np.fft.fft(datos_procesados[:int(N_fft*ratio)]))
    
    eje_w_in = np.linspace(-np.pi, np.pi, len(W_in))
    eje_w_out = np.linspace(-np.pi, np.pi, len(W_out))
    
    fig_w, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(eje_w_in, 20*np.log10(np.abs(W_in)+1e-9), 'k--', alpha=0.5, label='Original')
    ax3.plot(eje_w_out, 20*np.log10(np.abs(W_out)+1e-9), 'r-', label='Procesada')
    ax3.set_xlim(-np.pi, np.pi)
    ax3.set_xlabel(r"Frecuencia $\omega$ (rad)")
    ax3.set_xticks([-np.pi, 0, np.pi])
    ax3.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax3.legend()
    ax3.grid(alpha=0.3)
    st.pyplot(fig_w)

# --- DESCARGA Y REPRODUCCIÓN ---
st.divider()
col_sal1, col_sal2 = st.columns([3, 1])
with col_sal1:
    audio_final = np.nan_to_num(datos_procesados)
    pico = np.max(np.abs(audio_final))
    if pico > 0: audio_final /= pico
    audio_final = np.clip(audio_final, -1.0, 1.0)
    
    buffer = io.BytesIO()
    write(buffer, fs_salida, (audio_final * 32767).astype(np.int16))
    renderizar_reproductor(buffer, fs_salida, st.session_state.id_archivo)

with col_sal2:
    st.download_button("💾 Descargar WAV", buffer, "audio_procesado.wav", "audio/wav")

# Limpieza de memoria
del datos_remuestreados, datos_procesados, audio_final, buffer
gc.collect()
