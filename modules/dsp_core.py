import numpy as np
import soundfile as sf
import scipy.signal as signal # Únicamente para el motor de la ecuación en diferencias

# ==============================================================================
# MÓDULO DE PROCESAMIENTO DE SEÑALES (DSP CORE)
# Basado en la teoría de Sistemas Lineales e Invariantes en el Tiempo (LTI)
# ==============================================================================

def cargar_senal_audio(buffer_archivo):
    """
    Carga la señal discreta x[n] desde un archivo de audio.
    
    Procesamiento:
    1. Lectura de muestras.
    2. Conversión a monoaural (promedio de canales).
    3. Normalización de amplitud para garantizar |x[n]| <= 1.
    """
    try:
        x_n, fs = sf.read(buffer_archivo)
        
        # Si es estéreo, promediamos para obtener x[n] mono
        if len(x_n.shape) > 1:
            x_n = x_n.mean(axis=1)
            
        x_n = x_n.astype(np.float32)
        
        # Normalización (Scaling)
        max_amplitud = np.max(np.abs(x_n))
        if max_amplitud > 1e-6:
            x_n = x_n / max_amplitud
            
        return x_n, fs
    except:
        return np.zeros(100, dtype=np.float32), 44100

# ==========================================
# ANÁLISIS EN FRECUENCIA (CAP. 9 OPPENHEIM)
# ==========================================

def fft_diezmado_en_tiempo(x):
    """
    Implementación manual del algoritmo FFT (Transformada Rápida de Fourier).
    Utiliza la técnica de 'Diezmado en el Tiempo' (Decimation-in-Time) Radix-2.
    
    Base Teórica:
    Descompone la DFT de N puntos en dos DFTs de N/2 puntos (muestras pares e impares).
    X[k] = Par[k] + W_N^k * Impar[k]
    Donde W_N^k es el factor de giro (Twiddle Factor).
    """
    N = len(x)
    if N <= 1: return x
    
    # División (Divide y Vencerás)
    pares = fft_diezmado_en_tiempo(x[0::2])
    impares = fft_diezmado_en_tiempo(x[1::2])
    
    # Cálculo de Factores de Giro: W_N^k = e^(-j * 2*pi * k / N)
    k = np.arange(N // 2)
    W_N = np.exp(-2j * np.pi * k / N)
    
    # Combinación (Mariposa)
    t = W_N * impares
    X_k = np.concatenate([pares + t, pares - t])
    
    return X_k

def calcular_espectro_magnitud(x_n, fs):
    """
    Calcula la Magnitud del Espectro |X(jw)| o |X[k]| usando ventaneo.
    """
    # Seleccionamos un segmento representativo para la visualización (Ventaneo)
    # Esto equivale a multiplicar x[n] por una ventana rectangular o Hanning w[n].
    N_ventana = 2048 
    
    if len(x_n) > N_ventana:
        mitad = len(x_n) // 2
        segmento = x_n[mitad : mitad + N_ventana]
    else:
        # Zero-padding si la señal es muy corta
        sig_potencia_2 = 1 << (len(x_n) - 1).bit_length()
        segmento = np.pad(x_n, (0, sig_potencia_2 - len(x_n)))

    # Aplicación de Ventana Hanning para reducir la Fuga Espectral (Spectral Leakage)
    N = len(segmento)
    n = np.arange(N)
    w_n = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    
    # FFT Manual
    X_k = fft_diezmado_en_tiempo(segmento * w_n)
    magnitud = np.abs(X_k)
    
    # Eje de frecuencias (k * fs / N)
    frecuencias = np.fft.rfftfreq(N, d=1/fs) 
    
    # Retornamos solo la parte positiva del espectro (simetría conjugada de señales reales)
    mitad_N = N // 2 + 1
    return frecuencias[:mitad_N], magnitud[:mitad_N]

# ==========================================
# MUESTREO Y CONVOLUCIÓN (CAP. 4 Y 7 OPPENHEIM)
# ==========================================

def generar_respuesta_impulso_sinc(w_c_norm, L_taps):
    """
    Genera la respuesta al impulso h[n] de un Filtro Paso Bajo Ideal.
    
    Teoría:
    h[n] = [sin(wc * n) / (pi * n)] * w[n]
    Donde 'wc' es la frecuencia de corte angular y 'w[n]' es una ventana (Blackman)
    para truncar la respuesta infinita y hacer el filtro causal y realizable (FIR).
    """
    # Aseguramos simetría (Fase Lineal) con número impar de coeficientes
    if L_taps % 2 == 0: L_taps += 1
    
    n = np.arange(-(L_taps // 2), (L_taps // 2) + 1)
    
    # Función Sinc (sin(pi*x)/(pi*x)) normalizada
    # w_c_norm es la frecuencia de corte relativa a Nyquist
    h_n = np.sinc(w_c_norm * n)
    
    # Ventana Blackman para minimizar lóbulos laterales (Gibbs Phenomenon)
    ventana = np.blackman(len(n))
    h_n = h_n * ventana
    
    # Normalización de Energía: Suma(h[n]) = 1 para ganancia unitaria en DC
    suma = np.sum(h_n)
    if suma != 0:
        h_n /= suma
        
    return h_n

def conversion_tasa_muestreo(x_n, fs_original, M, L):
    """
    Implementación del Sistema de Conversión de Tasa de Muestreo.
    Referencia: Oppenheim Cap. 7 (Muestreo).
    
    Bloques:
    1. Expansor (Upsampling): x_e[n] (Inserta L-1 ceros).
    2. Filtro Paso Bajo h[n]: Interpolación y Anti-Solapamiento (Anti-Aliasing).
    3. Diezmador (Downsampling): x_d[n] (Toma 1 muestra cada M).
    """
    # Caso trivial (Bypass)
    if M == 1 and L == 1:
        return x_n, fs_original

    # 1. EXPANSIÓN (Upsampling) por factor L
    N = len(x_n)
    x_expandida = np.zeros(N * L, dtype=x_n.dtype)
    x_expandida[::L] = x_n # x_e[n] = x[n/L] si n es multiplo de L
    
    # 2. FILTRADO (Convolución Discreta)
    # Frecuencia de corte crítica: min(pi/L, pi/M) para evitar imágenes y aliasing.
    # Normalizamos respecto a Nyquist (1.0).
    w_corte_norm = 1.0 / max(L, M)
    
    # Longitud del filtro (Taps) proporcional a la exigencia del corte
    num_taps = 40 * max(L, M) + 1
    h_n = generar_respuesta_impulso_sinc(w_corte_norm, num_taps)
    
    # Compensación de ganancia por expansión (Conservación de energía)
    h_n *= L
    
    # y[n] = x[n] * h[n] (Suma de Convolución)
    # Usamos 'same' para mantener la referencia temporal centrada
    x_filtrada = np.convolve(x_expandida, h_n, mode='same')
    
    # 3. DIEZMADO (Downsampling) por factor M
    # y[n] = x_f[n * M]
    x_final = x_filtrada[::M]
    
    fs_nueva = int(fs_original * L / M)
    return x_final, fs_nueva

# ==========================================
# FILTRADO IIR Y SISTEMAS LTI (CAP. 5 OPPENHEIM)
# ==========================================

def disenar_coeficientes_diferencias(fc, fs, ganancia_db):
    """
    Calcula los coeficientes {bk} y {ak} de la Ecuación en Diferencias
    para un filtro de segundo orden (Biquad) tipo 'Peaking EQ'.
    
    Se basa en la Transformada Bilineal para mapear el plano S al plano Z.
    """
    # Frecuencia angular discreta w0
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / 2.0  # Q fijo en 1.0 para ancho de banda musical
    A = 10 ** (ganancia_db / 10.0)
    
    # Coeficientes del numerador (Feedforward) y denominador (Feedback)
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    # Normalización estándar (a0 = 1)
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    return b, a

def aplicar_ecuacion_diferencias(x_n, b, a):
    """
    Implementa el sistema LTI discreto definido por la Ecuación en Diferencias
    Lineal de Coeficientes Constantes:
    
    sum(ak * y[n-k]) = sum(bk * x[n-k])
    
    Nota: Se utiliza 'lfilter' como motor computacional eficiente de esta ecuación.
    """
    return signal.lfilter(b, a, x_n)

def sistema_ecualizador(x_n, fs, ganancias_bandas):
    """
    Sistema LTI compuesto por múltiples filtros en serie (cascada).
    Implementa protección contra violación del Teorema de Nyquist.
    """
    # Bypass si no hay ganancia (Respuesta plana)
    if all(abs(g) < 0.1 for g in ganancias_bandas.values()):
        return x_n

    frecuencias_centrales = {
        "Sub-Bass": 40, "Bass": 150, "Low Mids": 1000,
        "High Mids": 3000, "Presence": 5000, "Brilliance": 10000
    }
    
    y_n = x_n.copy()
    limite_nyquist = fs / 2.0
    
    for banda, ganancia in ganancias_bandas.items():
        if abs(ganancia) > 0.1:
            fc_teorica = frecuencias_centrales.get(banda, 1000)
            
            # --- PROTECCIÓN DE NYQUIST ---
            # Si fc >= fs/2, el filtro es irrealizable digitalmente.
            # Ajustamos fc para que quede dentro del círculo unitario (estabilidad).
            techo_seguro = limite_nyquist * 0.90
            
            if fc_teorica >= techo_seguro:
                # Ajuste dinámico de frecuencia ("Frequency Clamping")
                fc_efectiva = techo_seguro
            else:
                fc_efectiva = fc_teorica
            
            # Si tras el ajuste la frecuencia es válida (> 10 Hz), aplicamos el filtro
            if fc_efectiva > 10:
                b, a = disenar_coeficientes_diferencias(fc_efectiva, fs, ganancia)
                y_n = aplicar_ecuacion_diferencias(y_n, b, a)
            
    # Saturación suave para evitar overflow numérico
    return np.clip(y_n, -1.0, 1.0)
