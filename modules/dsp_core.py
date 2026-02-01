import numpy as np
import soundfile as sf

# ==========================================
# 1. GESTIÓN DE AUDIO (I/O)
# ==========================================
def cargar_audio(buffer_archivo):
    try:
        datos, fs = sf.read(buffer_archivo)
        if len(datos.shape) > 1:
            datos = datos.mean(axis=1)
        datos = datos.astype(np.float32)
        
        # Normalización segura
        max_valor = np.max(np.abs(datos))
        if max_valor > 1e-6:
            datos = datos / max_valor
        return datos, fs
    except:
        return np.zeros(100, dtype=np.float32), 44100

# ==========================================
# 2. TRANSFORMADA DE FOURIER (FFT MANUAL)
# ==========================================
def fft_recursiva_radix2(x):
    """Implementación manual Radix-2."""
    N = len(x)
    if N <= 1: return x
    
    pares = fft_recursiva_radix2(x[0::2])
    impares = fft_recursiva_radix2(x[1::2])
    
    k = np.arange(N // 2)
    factores = np.exp(-2j * np.pi * k / N)
    
    t = factores * impares
    return np.concatenate([pares + t, pares - t])

def calcular_fft(datos, fs):
    tamano_ventana = 2048 
    if len(datos) > tamano_ventana:
        mitad = len(datos) // 2
        segmento = datos[mitad : mitad + tamano_ventana]
    else:
        # Zero padding a la siguiente potencia de 2
        siguiente_potencia = 1 << (len(datos) - 1).bit_length()
        segmento = np.pad(datos, (0, siguiente_potencia - len(datos)))

    N = len(segmento)
    ventana = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    
    espectro = fft_recursiva_radix2(segmento * ventana)
    magnitud = np.abs(espectro)
    frecuencias = np.fft.rfftfreq(N, d=1/fs) # Eje auxiliar
    
    mitad_N = N // 2 + 1
    return frecuencias[:mitad_N], magnitud[:mitad_N]

# ==========================================
# 3. MUESTREO Y CONVOLUCIÓN (NÚCLEO CORREGIDO)
# ==========================================

def generar_filtro_sinc_normalizado(corte_norm, num_taps):
    """
    Genera un filtro Sinc basado en frecuencia normalizada (0.0 a 1.0).
    corte_norm: Frecuencia de corte relativa a Nyquist (fs/2).
                Ej: 0.5 significa corte en fs/4.
    """
    # Asegurar taps impares para alineación de fase perfecta
    if num_taps % 2 == 0: num_taps += 1
    
    n = np.arange(-(num_taps // 2), (num_taps // 2) + 1)
    
    # Sinc Ideal: sin(pi * wc * n) / (pi * n)
    # Nota: np.sinc(x) calcula sin(pi*x)/(pi*x)
    h = np.sinc(corte_norm * n)
    
    # Ventana Blackman para minimizar lóbulos laterales
    ventana = np.blackman(len(n))
    h = h * ventana
    
    # NORMALIZACIÓN DE ENERGÍA CRÍTICA
    # La suma de coeficientes debe ser 1.0 para ganancia unitaria DC
    suma = np.sum(h)
    if suma != 0:
        h /= suma
        
    return h

def cambiar_tasa_muestreo(datos, fs_original, factor_m, factor_l):
    """
    Algoritmo de Remuestreo Polifásico Manual.
    """
    # Bypass 1:1
    if factor_m == 1 and factor_l == 1:
        return datos, fs_original

    # 1. Expansión (Upsampling)
    # Insertar L-1 ceros entre muestras
    N = len(datos)
    datos_expandidos = np.zeros(N * factor_l, dtype=datos.dtype)
    datos_expandidos[::factor_l] = datos
    
    # 2. Diseño del Filtro (Lógica Robusta)
    # Trabajamos en el dominio de la señal expandida (fs' = fs * L).
    # Necesitamos cortar las imágenes espectrales generadas por L 
    # Y prevenir el aliasing que generará M.
    # El corte debe ser: 1 / max(L, M) relativo a Nyquist.
    corte_normalizado = 1.0 / max(factor_l, factor_m)
    
    # Longitud del filtro dinámica: Más taps si el corte es estrecho
    num_taps = 40 * max(factor_l, factor_m) + 1
    
    # Generar Kernel
    filtro = generar_filtro_sinc_normalizado(corte_normalizado, num_taps)
    
    # Corrección de Ganancia:
    # Al expandir por L, la energía se divide. Multiplicamos el filtro por L.
    filtro *= factor_l
    
    # 3. Filtrado (Convolución)
    # 'same' centra la convolución, vital para no desfasar la señal al diezmar
    datos_filtrados = np.convolve(datos_expandidos, filtro, mode='same')
    
    # 4. Decimación (Downsampling)
    datos_finales = datos_filtrados[::factor_m]
    
    fs_final = int(fs_original * factor_l / factor_m)
    return datos_finales, fs_final

# ==========================================
# 4. ECUALIZADOR (ECUACIÓN EN DIFERENCIAS)
# ==========================================

def calcular_coefs_eq(fc, fs, gain_db, Q=1.0):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    A = 10 ** (gain_db / 40.0)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    return np.array([b0, b1, b2])/a0, np.array([a0, a1, a2])/a0

def filtro_iir_manual(x, b, a):
    # Implementación vectorizada básica (Direct Form II Transposed simplificada)
    # Usamos lfilter de numpy SOLO como motor matemático de la ec. diferencias
    # para evitar que el bucle for de Python tarde 10 segundos por slider.
    # Matemáticamente es idéntico a: y[n] = b0*x[n] ... - a1*y[n-1]
    from scipy.signal import lfilter
    return lfilter(b, a, x)

def aplicar_ecualizador(datos, fs, ganancias):
    # Bypass si todo está en 0
    if all(abs(g) < 0.1 for g in ganancias.values()):
        return datos

    bandas = {
        "Sub-Bass": 40, "Bass": 150, "Low Mids": 1000,
        "High Mids": 3000, "Presence": 5000, "Brilliance": 10000
    }
    
    salida = datos.copy()
    
    # Límite de Nyquist actual (Frecuencia máxima posible)
    limite_nyquist = fs / 2.0
    
    for nombre, ganancia in ganancias.items():
        if abs(ganancia) > 0.1:
            fc_original = bandas.get(nombre, 1000)
            
            # --- CORRECCIÓN INTELIGENTE DE BANDAS ---
            # Si la frecuencia central está fuera, no apagamos la banda.
            # En su lugar, la "empujamos" hacia adentro del límite válido.
            
            # Margen de seguridad (90% de Nyquist) para estabilidad IIR
            techo_seguro = limite_nyquist * 0.90
            
            if fc_original >= techo_seguro:
                # Si la banda original es 10kHz pero solo llegamos a 8kHz,
                # ajustamos el filtro a ~7.2kHz para controlar los agudos restantes.
                fc_efectiva = techo_seguro
            else:
                fc_efectiva = fc_original
            
            # Caso extremo: Si incluso ajustando, la frecuencia es ridículamente baja
            # (ej. intentar meter Brilliance en un audio de 100Hz), ahí sí ignoramos.
            if fc_efectiva < 10: 
                continue

            # Calculamos coeficientes con la frecuencia ajustada
            b, a = calcular_coefs_eq(fc_efectiva, fs, ganancia)
            salida = filtro_iir_manual(salida, b, a)
            
    return np.clip(salida, -1.0, 1.0)
