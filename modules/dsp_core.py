import numpy as np
import soundfile as sf
import scipy.signal as signal

# ==========================================
# GESTIÓN DE AUDIO
# ==========================================

def load_audio(file_buffer):
    """Carga el audio, convierte a mono y normaliza."""
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
    """Calcula la magnitud del espectro usando FFT."""
    n = len(data)
    if n == 0: return [0], [0]
    window = np.hanning(n)
    freq = np.fft.rfftfreq(n, d=1/fs)
    magnitude = np.abs(np.fft.rfft(data * window))
    return freq, magnitude

# ==========================================
# TEORÍA DE MUESTREO (Manual / Oppenheim Cap. 7)
# ==========================================

def generate_manual_sinc_kernel(cutoff, fs, num_taps=101):
    """
    Genera manualmente la respuesta al impulso h[n] de un filtro paso bajo ideal.
    Teoría: h[n] = Sinc(w_c * n) * Ventana
    """
    n = np.arange(-num_taps // 2, num_taps // 2 + 1)
    # Frecuencia normalizada respecto a Nyquist
    fc = cutoff / (fs / 2)
    # np.sinc(x) es sin(pi*x)/(pi*x)
    h = np.sinc(fc * n) 
    # Ventana Hamming para suavizar bordes (Filtro FIR)
    window = np.hamming(len(n))
    h = h * window
    # Normalizar energía
    h = h / np.sum(h)
    return h

def manual_upsample(x, L):
    """Expansión: Inserta L-1 ceros entre muestras (Oppenheim Cap 7.1)."""
    if L == 1: return x
    N = len(x)
    y = np.zeros(N * L, dtype=x.dtype)
    y[::L] = x
    return y

def manual_downsample(x, M):
    """Decimación: Toma una muestra cada M (Oppenheim Cap 7.1)."""
    if M == 1: return x
    return x[::M]

def change_sampling_rate(data, original_fs, m_factor, l_factor):
    """
    Cadena completa: Upsampler -> Filtro LP -> Downsampler
    """
    # --- TRUE BYPASS (Solución a tu TOC) ---
    if m_factor == 1 and l_factor == 1:
        return data, original_fs
    # ---------------------------------------

    # 1. Expansión (Upsampling)
    x_expanded = manual_upsample(data, l_factor)
    
    # 2. Diseño del Filtro de Interpolación / Anti-Aliasing
    new_fs_temp = original_fs * l_factor
    # La frecuencia de corte debe ser la menor entre Nyquist original y Nyquist destino
    cutoff_freq = min(original_fs / 2, (new_fs_temp / m_factor) / 2)
    
    # Generar h[n] manualmente
    # Ganancia * L requerida por la pérdida de energía al insertar ceros
    h = generate_manual_sinc_kernel(cutoff_freq, new_fs_temp, num_taps=61) * l_factor
    
    # 3. Convolución Discreta (Filtrado)
    # y[n] = x[n] * h[n]
    x_filtered = np.convolve(x_expanded, h, mode='same')
    
    # 4. Decimación (Downsampling)
    x_final = manual_downsample(x_filtered, m_factor)
    
    final_fs = int(original_fs * l_factor / m_factor)
    
    return x_final, final_fs

# ==========================================
# ECUALIZADOR (Sistemas LTI / Oppenheim Cap. 5)
# ==========================================

def get_difference_equation_coeffs(low, high, fs):
    """Obtiene coeficientes a_k, b_k para la ecuación en diferencias."""
    nyquist = 0.5 * fs
    if low >= nyquist: return None, None
    if high >= nyquist: high = nyquist * 0.99
    
    # Diseño Butterworth (Prototipo analógico -> Bilineal -> Digital)
    b, a = signal.butter(2, [low, high], btype='band', fs=fs)
    return b, a

def apply_equalizer(data, fs, gains_db):
    """
    Ecualizador Gráfico implementado como suma de sistemas LTI.
    """
    # --- TRUE BYPASS (Solución a tu TOC) ---
    # Si todas las ganancias son casi cero (< 0.1 dB), devolvemos la señal pura.
    # Esto evita problemas de fase y asegura que Entrada == Salida.
    if all(abs(g) < 0.1 for g in gains_db.values()):
        return data
    # ---------------------------------------

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
        b, a = get_difference_equation_coeffs(low, high, fs)
        
        if b is not None:
            filter_active = True
            # Implementación de la Ecuación en Diferencias:
            # sum(a_k * y[n-k]) = sum(b_k * x[n-k])
            # Usamos lfilter como motor eficiente de esta ecuación.
            band_signal = signal.lfilter(b, a, data)
            
            # Ganancia Lineal
            gain = 10 ** (gains_db.get(band_name, 0) / 20.0)
            
            # Suma (Superposición)
            output_signal += band_signal * gain

    if not filter_active:
        return data

    return output_signal
