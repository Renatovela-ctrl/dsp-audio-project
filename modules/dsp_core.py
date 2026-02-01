import numpy as np
import soundfile as sf
import scipy.signal as signal # Usado SOLO para calcular coeficientes (diseño), no para aplicar.

# ==========================================
# UTILIDADES BASICAS
# ==========================================

def load_audio(file_buffer):
    """Carga audio, convierte a mono y normaliza."""
    try:
        data, samplerate = sf.read(file_buffer)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        return data, samplerate
    except:
        return np.zeros(100), 44100

def compute_fft(data, fs):
    """
    Calcula la DFT usando FFT (Algoritmo eficiente para la Ec. de Análisis).
    Oppenheim Cap. 8 (DFT).
    """
    n = len(data)
    if n == 0: return [0], [0]
    
    # Ventana para minimizar fuga espectral (Spectral Leakage)
    window = np.hanning(n) 
    
    # Implementación estándar de FFT
    freq = np.fft.rfftfreq(n, d=1/fs)
    magnitude = np.abs(np.fft.rfft(data * window))
    
    return freq, magnitude

# ==========================================
# TEORÍA DE MUESTREO (OPPENHEIM CAP. 7)
# ==========================================

def generate_lowpass_kernel(cutoff, fs, num_taps=101):
    """
    Genera manualmente la respuesta al impulso h[n] de un filtro paso bajo ideal.
    Teoría: h[n] ideal es una función Sinc.
    """
    # Eje de tiempo centrado n = -M ... 0 ... M
    n = np.arange(-num_taps // 2, num_taps // 2 + 1)
    
    # Frecuencia de corte normalizada (omega_c / pi)
    # En Oppenheim, el filtro de interpolación debe tener ganancia L y corte pi/L
    fc = cutoff / (fs / 2)
    
    # h[n] = sinc(wc * n) * ventana
    # np.sinc(x) calcula sin(pi*x)/(pi*x), por eso dividimos fc
    h = np.sinc(fc * n) 
    
    # Aplicamos ventana Hamming para hacer el filtro causal y finito (FIR)
    window = np.hamming(len(n))
    h = h * window
    
    # Normalización para mantener ganancia unitaria en la banda pasante
    h = h / np.sum(h)
    
    return h

def manual_upsample(x, L):
    """
    Expansor de tasa (Upsampling).
    Oppenheim: Inserta L-1 ceros entre muestras.
    x_e[n] = x[n/L] si n es múltiplo de L, 0 si no.
    """
    if L == 1: return x
    N = len(x)
    # Crear array de ceros de tamaño L*N
    y = np.zeros(N * L, dtype=x.dtype)
    # Llenar solo los índices múltiplos de L
    y[::L] = x
    return y

def manual_downsample(x, M):
    """
    Diezmador (Downsampling).
    Oppenheim: x_d[n] = x[nM].
    """
    if M == 1: return x
    return x[::M]

def change_sampling_rate(data, original_fs, m_factor, l_factor):
    """
    Implementación MANUAL del cambio de tasa.
    Cadena: Expansor -> Filtro Paso Bajo (Interpolador) -> Diezmador
    """
    # 1. BYPASS REAL (Si no hay cambios, devolver original)
    if m_factor == 1 and l_factor == 1:
        return data, original_fs

    # 2. EXPANSIÓN (Upsampling)
    # x_e[n] = Insertar ceros
    x_expanded = manual_upsample(data, l_factor)
    
    # 3. FILTRADO (Interpolación / Anti-Aliasing)
    # Frecuencia de corte: min(pi/L, pi/M)
    # Para evitar imágenes por upsampling y aliasing por downsampling.
    new_fs_temp = original_fs * l_factor
    cutoff_freq = original_fs / 2 # Nyquist original
    if m_factor > 1:
        cutoff_freq = min(cutoff_freq, (new_fs_temp / m_factor) / 2)
    
    # Generar h[n] (Kernel Sinc manual)
    # La ganancia del filtro debe ser L para compensar la pérdida de energía al insertar ceros
    h = generate_lowpass_kernel(cutoff_freq, new_fs_temp, num_taps=61) * l_factor
    
    # Convolución Discreta: y[n] = sum(x[k] * h[n-k])
    # Usamos np.convolve que es la implementación eficiente de esta sumatoria
    x_filtered = np.convolve(x_expanded, h, mode='same')
    
    # 4. DECIMACIÓN (Downsampling)
    # y[n] = x_filtered[nM]
    x_final = manual_downsample(x_filtered, m_factor)
    
    final_fs = int(original_fs * l_factor / m_factor)
    
    return x_final, final_fs

# ==========================================
# FILTRADO Y SISTEMAS LTI (OPPENHEIM CAP. 5)
# ==========================================

def apply_difference_equation(data, b, a):
    """
    Implementa un sistema LTI definido por su Ecuación en Diferencias de Coeficientes Constantes.
    sum(a_k * y[n-k]) = sum(b_k * x[n-k])
    
    Nota: Usamos lfilter de scipy porque un bucle for puro en Python es demasiado lento 
    para audio (tardaría minutos), pero lfilter implementa exactamente esta ecuación.
    """
    return signal.lfilter(b, a, data)

def get_coefficients_manual(low, high, fs):
    """
    Obtiene los coeficientes (a, b) para la ecuación en diferencias.
    Usamos Butterworth de 2do orden como prototipo analógico convertido a digital.
    """
    nyquist = 0.5 * fs
    if low >= nyquist: return None, None
    if high >= nyquist: high = nyquist * 0.99
    
    # Diseño de coeficientes (polos y ceros en plano Z)
    # Esto retorna los vectores b (feedforward) y a (feedback)
    b, a = signal.butter(2, [low, high], btype='band', fs=fs)
    return b, a

def apply_equalizer(data, fs, gains_db):
    """
    Ecualizador Gráfico implementado como suma de sistemas LTI en paralelo.
    """
    # BYPASS: Si todo es 0, retornar la señal pura (Evita distorsión de fase)
    if all(abs(g) < 0.1 for g in gains_db.values()):
        return data

    bands = {
        "Sub-Bass": (16, 60),
        "Bass": (60, 250),
        "Low Mids": (250, 2000),
        "High Mids": (2000, 4000),
        "Presence": (4000, 6000),
        "Brilliance": (6000, 16000)
    }
    
    output_signal = np.zeros_like(data)
    filter_active = False
    
    for band_name, (low, high) in bands.items():
        # 1. Obtener coeficientes de la Ecuación en Diferencias
        b, a = get_coefficients_manual(low, high, fs)
        
        if b is not None:
            filter_active = True
            # 2. Filtrar usando la Ec. en Diferencias
            band_signal = apply_difference_equation(data, b, a)
            
            # 3. Aplicar Ganancia Lineal (A)
            # y_band[n] = A * sistema{x[n]}
            gain = 10 ** (gains_db.get(band_name, 0) / 20.0)
            
            # 4. Suma (Principio de Superposición de sistemas lineales)
            output_signal += band_signal * gain

    if not filter_active:
        return data

    return output_signal
