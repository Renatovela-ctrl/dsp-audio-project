import numpy as np
import soundfile as sf
import scipy.signal as signal

# ==========================================
# 1. GESTIÓN DE AUDIO ROBUSTA
# ==========================================
def load_audio(file_buffer):
    try:
        data, samplerate = sf.read(file_buffer)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        # Convertir a float32 para evitar overflow de int16
        data = data.astype(np.float32)
        
        # Normalización segura
        max_val = np.max(np.abs(data))
        if max_val > 1e-6: # Evitar división por cero si es silencio
            data = data / max_val
            
        return data, samplerate
    except:
        return np.zeros(100, dtype=np.float32), 44100

def compute_fft(data, fs):
    n = len(data)
    if n == 0: return np.array([1.0]), np.array([0.0])
    
    window = np.hanning(n)
    # FFT Real (más eficiente para audio)
    freq = np.fft.rfftfreq(n, d=1/fs)
    magnitude = np.abs(np.fft.rfft(data * window))
    
    return freq, magnitude

# ==========================================
# 2. MUESTREO (Manual - Oppenheim Cap. 7)
# ==========================================
def generate_manual_sinc_kernel(cutoff, fs, num_taps=101):
    """Genera filtro FIR Sinc ventaneado manualmente."""
    if fs <= 0: return np.ones(1)
    
    n = np.arange(-num_taps // 2, num_taps // 2 + 1)
    fc = cutoff / (fs / 2) # Frecuencia normalizada (0 a 1)
    
    # Sinc Ideal: sin(pi*x)/(pi*x)
    h = np.sinc(fc * n) 
    
    # Ventana Hamming (Suavizado)
    window = np.hamming(len(n))
    h = h * window
    
    # Normalizar ganancia unitaria
    h_sum = np.sum(h)
    if h_sum != 0:
        h /= h_sum
        
    return h

def change_sampling_rate(data, original_fs, m_factor, l_factor):
    # --- TRUE BYPASS ---
    if m_factor == 1 and l_factor == 1:
        return data, original_fs
        
    # Validaciones de seguridad
    if m_factor < 1: m_factor = 1
    if l_factor < 1: l_factor = 1
    if len(data) == 0: return data, original_fs

    # 1. Expansión (Upsampling) -> Insertar ceros
    N = len(data)
    x_expanded = np.zeros(N * l_factor, dtype=data.dtype)
    x_expanded[::l_factor] = data
    
    # 2. Filtro Anti-Imagen / Anti-Aliasing
    new_fs_temp = original_fs * l_factor
    cutoff_freq = min(original_fs / 2, (new_fs_temp / m_factor) / 2)
    
    # Generar Kernel (ganancia * L para recuperar energía)
    h = generate_manual_sinc_kernel(cutoff_freq, new_fs_temp, num_taps=61) * l_factor
    
    # Convolución
    x_filtered = np.convolve(x_expanded, h, mode='same')
    
    # 3. Decimación (Downsampling)
    x_final = x_filtered[::m_factor]
    
    final_fs = int(original_fs * l_factor / m_factor)
    return x_final, final_fs

# ==========================================
# 3. ECUALIZADOR (Estable / Oppenheim Cap. 5)
# ==========================================
def get_stable_filter_coeffs(low, high, fs):
    """
    Diseña filtros usando SOS (Second Order Sections).
    La forma 'ba' (Ec. Diferencias directa) es INESTABLE para frecuencias bajas 
    y causa explosiones numéricas (10^60). SOS es la solución de ingeniería.
    """
    nyquist = 0.5 * fs
    
    # Protecciones de límites
    if low <= 0: low = 1  # Evitar 0 Hz
    if low >= nyquist: return None # Fuera de rango
    if high >= nyquist: high = nyquist * 0.99 # Evitar tocar Nyquist
    if high <= low: return None # Cruce inválido

    try:
        # 'sos' descompone el filtro en cascada de secciones de 2do orden estables
        sos = signal.butter(2, [low, high], btype='band', fs=fs, output='sos')
        return sos
    except:
        return None

def apply_equalizer(data, fs, gains_db):
    # --- TRUE BYPASS ---
    # Si todo es 0dB, retornar original para fidelidad perfecta
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
        sos = get_stable_filter_coeffs(low, high, fs)
        
        if sos is not None:
            filter_active = True
            # Aplicar filtro SOS (Matemáticamente equivalente a Ec. Diferencias pero estable)
            band_signal = signal.sosfilt(sos, data)
            
            # Ganancia
            gain = 10 ** (gains_db.get(band_name, 0) / 20.0)
            output_signal += band_signal * gain

    if not filter_active:
        return data

    # Sanitización final: Eliminar posibles infinitos si algo falló
    output_signal = np.nan_to_num(output_signal, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Clipping duro para seguridad auditiva [-1.0, 1.0]
    output_signal = np.clip(output_signal, -1.0, 1.0)
    
    return output_signal
