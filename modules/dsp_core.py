import numpy as np
import scipy.signal as signal

def load_audio(file_buffer):
    """Carga el audio y lo normaliza."""
    import soundfile as sf
    data, samplerate = sf.read(file_buffer)
    # Si es estéreo, convertir a mono para simplificar el procesamiento DSP
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return data, samplerate

def change_sampling_rate(data, original_fs, m_factor, l_factor):
    """
    Realiza la conversión de tasa de muestreo (Resampling).
    M: Factor de Decimación (Downsampling)
    L: Factor de Expansión (Upsampling/Interpolation)
    """
    # La nueva tasa de muestreo teórica
    new_fs = int(original_fs * l_factor / m_factor)
    
    # resample_poly aplica eficientemente el filtro FIR polifásico
    # Esto cumple con el requisito de filtrar para evitar aliasing o imágenes espectrales
    resampled_data = signal.resample_poly(data, l_factor, m_factor)
    
    return resampled_data, new_fs

def design_bandpass_filter(lowcut, highcut, fs, order=4):
    """Diseña un filtro pasabanda Butterworth."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Verificación de seguridad para evitar errores cerca de Nyquist
    if high >= 1.0:
        high = 0.99
        
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_equalizer(data, fs, gains_db):
    """
    Aplica un ecualizador gráfico de 6 bandas.
    gains_db: Diccionario con las ganancias en dB para cada banda.
    """
    # Definición de bandas según el documento [cite: 123-128]
    bands = {
        "Sub-Bass": (16, 60),
        "Bass": (60, 250),
        "Low Mids": (250, 2000),
        "High Mids": (2000, 4000),
        "Presence": (4000, 6000),
        "Brilliance": (6000, 16000)
    }
    
    output_signal = np.zeros_like(data)
    
    for band_name, (low, high) in bands.items():
        # 1. Diseñar filtro para la banda
        b, a = design_bandpass_filter(low, high, fs)
        
        # 2. Filtrar la señal original para aislar esta banda
        band_signal = signal.lfilter(b, a, data)
        
        # 3. Aplicar ganancia
        # Convertir dB a factor lineal: 10^(dB/20)
        gain_linear = 10 ** (gains_db[band_name] / 20.0)
        
        # 4. Sumar al output general
        output_signal += band_signal * gain_linear
        
    return output_signal

def compute_fft(data, fs):
    """Calcula la FFT para visualización espectral."""
    n = len(data)
    freq = np.fft.rfftfreq(n, d=1/fs)
    magnitude = np.abs(np.fft.rfft(data))
    return freq, magnitude
