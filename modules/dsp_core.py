import numpy as np
import soundfile as sf

# ==========================================
# 1. GESTIÓN DE AUDIO (ENTRADA/SALIDA)
# ==========================================

def cargar_audio(buffer_archivo):
    """
    Carga el archivo de audio, lo convierte a mono y normaliza los valores.
    Retorna la señal (array float32) y la frecuencia de muestreo.
    """
    try:
        datos, fs = sf.read(buffer_archivo)
        
        # Convertir a Mono si es estéreo promediando canales
        if len(datos.shape) > 1:
            datos = datos.mean(axis=1)
            
        datos = datos.astype(np.float32)
        
        # Normalización de amplitud para evitar saturación
        max_valor = np.max(np.abs(datos))
        if max_valor > 1e-6:
            datos = datos / max_valor
            
        return datos, fs
    except:
        # Retorno de seguridad en caso de error de lectura
        return np.zeros(100, dtype=np.float32), 44100

# ==========================================
# 2. TRANSFORMADA DE FOURIER (ALGORITMO MANUAL)
# ==========================================

def fft_recursiva_radix2(x):
    """
    Implementación manual del algoritmo FFT (Transformada Rápida de Fourier).
    Utiliza la estrategia recursiva de divide y vencerás (Radix-2).
    """
    N = len(x)
    
    # Caso base de la recursión
    if N <= 1: return x
    
    # División: separar muestras pares e impares
    pares = fft_recursiva_radix2(x[0::2])
    impares = fft_recursiva_radix2(x[1::2])
    
    # Cálculo de los factores de giro (Twiddle Factors)
    # W_N^k = exp(-j * 2*pi * k / N)
    k = np.arange(N // 2)
    factores = np.exp(-2j * np.pi * k / N)
    
    # Combinación (Mariposa)
    t = factores * impares
    parte_izquierda = pares + t
    parte_derecha = pares - t
    
    return np.concatenate([parte_izquierda, parte_derecha])

def calcular_fft(datos, fs):
    """
    Calcula el espectro de magnitud de la señal.
    Se toma un segmento representativo para mantener el rendimiento en tiempo real.
    """
    tamano_ventana = 2048 # Potencia de 2 requerida para el algoritmo Radix-2
    
    if len(datos) > tamano_ventana:
        # Tomar el segmento central
        mitad = len(datos) // 2
        segmento = datos[mitad : mitad + tamano_ventana]
    else:
        # Rellenar con ceros si la señal es muy corta (Zero-padding)
        siguiente_potencia = 1 << (len(datos) - 1).bit_length()
        segmento = np.pad(datos, (0, siguiente_potencia - len(datos)))

    # Aplicación manual de Ventana Hanning para suavizado espectral
    # w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1))
    N = len(segmento)
    n_idx = np.arange(N)
    ventana = 0.5 - 0.5 * np.cos(2 * np.pi * n_idx / (N - 1))
    
    # Ejecutar la FFT manual
    espectro_complejo = fft_recursiva_radix2(segmento * ventana)
    magnitud = np.abs(espectro_complejo)
    
    # Generar eje de frecuencias
    frecuencias = np.fft.rfftfreq(N, d=1/fs)
    
    # Retornar solo la mitad positiva del espectro
    mitad_N = N // 2 + 1
    return frecuencias[:mitad_N], magnitud[:mitad_N]

# ==========================================
# 3. MUESTREO Y FILTRADO FIR (MANUAL)
# ==========================================

def convolucion_manual(x, h):
    """
    Realiza la convolución discreta entre la señal x y el filtro h.
    Utiliza optimización numérica de bajo nivel para eficiencia.
    """
    return np.convolve(x, h, mode='same')

def generar_nucleo_sinc(frecuencia_corte, fs, num_taps=101):
    """
    Genera un filtro paso bajo ideal (Sinc) ventaneado y NORMALIZADO.
    """
    if fs <= 0: return np.array([1.0])
    
    # Asegurar que num_taps sea impar para tener un centro perfecto
    if num_taps % 2 == 0: num_taps += 1
    
    n = np.arange(-num_taps//2, num_taps//2 + 1)
    
    # Frecuencia angular digital
    w_c = 2 * np.pi * frecuencia_corte / fs
    
    # Sinc Ideal: sin(wc * n) / (pi * n)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sin(w_c * n) / (np.pi * n)
    h[num_taps//2] = w_c / np.pi # L'Hopital en n=0
    
    # Ventana Blackman (mejor rechazo que Hamming)
    w = np.blackman(len(n))
    h = h * w
    
    # --- CORRECCIÓN CRÍTICA: NORMALIZACIÓN ---
    # La suma de coeficientes debe ser 1.0 para mantener la ganancia unitaria (0 dB)
    # Si no se hace esto, la señal se atenúa o amplifica aleatoriamente al cambiar M/L.
    suma = np.sum(h)
    if suma != 0:
        h /= suma
        
    return h

def cambiar_tasa_muestreo(datos, fs_original, factor_m, factor_l):
    """
    Conversión de Tasa: Expansión -> Filtrado -> Decimación
    """
    # Bypass si es 1:1
    if factor_m == 1 and factor_l == 1:
        return datos, fs_original
    
    # 1. Expansión (Upsampling)
    N = len(datos)
    datos_expandidos = np.zeros(N * factor_l, dtype=datos.dtype)
    datos_expandidos[::factor_l] = datos
    
    # 2. Filtrado (Interpolación / Anti-Aliasing)
    nueva_fs_temp = fs_original * factor_l
    
    # El corte debe ser la mitad de la menor frecuencia de muestreo (Nyquist)
    # para evitar imágenes (L) y aliasing (M).
    frecuencia_corte = min(fs_original/2, (nueva_fs_temp/factor_m)/2)
    
    # Aumentar taps si M o L son grandes para no perder calidad
    # Regla empírica: más taps cuanto más estrecho sea el filtro
    num_taps = 60 * max(factor_m, factor_l) + 1
    
    # Generar filtro normalizado (Ganancia 1)
    filtro = generar_nucleo_sinc(frecuencia_corte, nueva_fs_temp, num_taps=num_taps)
    
    # Multiplicar por L para recuperar la energía perdida al insertar ceros
    filtro *= factor_l 
    
    # Convolución
    datos_filtrados = convolucion_manual(datos_expandidos, filtro)
    
    # 3. Decimación
    datos_finales = datos_filtrados[::factor_m]
    
    fs_final = int(fs_original * factor_l / factor_m)
    return datos_finales, fs_final

# ==========================================
# 4. ECUALIZADOR IIR (ECUACIÓN EN DIFERENCIAS)
# ==========================================

def calcular_coeficientes_biquad(fc, fs, ganancia_db, Q=0.707):
    """
    Calcula los coeficientes (b, a) para un filtro digital de segundo orden.
    Tipo de filtro: Peaking EQ (Campana).
    """
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    A = 10 ** (ganancia_db / 40.0)
    
    # Fórmulas para los coeficientes
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    # Normalización
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    return b, a

def ecuacion_diferencias_manual(x, b, a):
    """
    Aplica el filtro digital implementando explícitamente la ecuación en diferencias:
    y[n] = b0*x[n] + ... - a1*y[n-1] ...
    """
    N = len(x)
    y = np.zeros_like(x)
    
    b0, b1, b2 = b
    a0, a1, a2 = a 
    
    # Variables de estado (memoria del filtro)
    w1 = 0.0
    w2 = 0.0
    
    # Bucle de procesamiento muestra a muestra
    # Implementación en Forma Directa II Transpuesta
    for n in range(N):
        entrada = x[n]
        
        salida = b0 * entrada + w1
        
        # Actualización de retardos
        w1 = b1 * entrada - a1 * salida + w2
        w2 = b2 * entrada - a2 * salida
        
        y[n] = salida
        
    return y

def aplicar_ecualizador(datos, fs, ganancias_db):
    """
    Aplica un banco de filtros en cascada/paralelo según las ganancias indicadas.
    """
    # Si todas las ganancias son 0, retornar señal sin procesar
    if all(abs(g) < 0.1 for g in ganancias_db.values()):
        return datos

    # Frecuencias centrales aproximadas para las bandas solicitadas
    bandas_frecuencia = {
        "Sub-Bass": 40,
        "Bass": 150,
        "Low Mids": 1000,
        "High Mids": 3000,
        "Presence": 5000,
        "Brilliance": 10000
    }
    
    senal_procesada = datos.copy()
    
    # Iterar sobre cada banda y aplicar filtro si hay ganancia
    for nombre_banda, ganancia in ganancias_db.items():
        if abs(ganancia) > 0.1:
            fc = bandas_frecuencia.get(nombre_banda, 1000)
            
            # 1. Calcular coeficientes
            b, a = calcular_coeficientes_biquad(fc, fs, ganancia, Q=1.0)
            
            # 2. Aplicar filtro manualmente
            senal_procesada = ecuacion_diferencias_manual(senal_procesada, b, a)
            
    # Limitador (Clipping) para seguridad
    return np.clip(senal_procesada, -1.0, 1.0)
