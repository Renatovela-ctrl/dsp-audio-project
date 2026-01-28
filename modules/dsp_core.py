import numpy as np
import scipy.signal as signal
import soundfile as sf

def load_audio(file_buffer):
    """
    Carga el audio de manera segura y lo convierte a mono y float32 normalizado.
    """
    try:
        data, samplerate = sf.read(file_buffer)
        
        # 1. Convertir a Mono si es estéreo
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        # 2. Asegurar tipo de dato float32 (estándar para DSP)
        data = data.astype(np.float32)

        # 3. Normalizar entrada [-1, 1]
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
            
        return data, samplerate
    except Exception as e:
        return np.zeros(100), 44100 # Retorno de emergencia

def change_sampling_rate(data, original_fs, m_factor, l_factor):
    """
    Conversión de tasa usando filtro polifásico (Antialiasing incluido).
    """
    # Protección contra valores absurdos
    if m_factor < 1: m_factor = 1
    if l_factor < 1: l_factor = 1

    # Nueva frecuencia teórica
    new_fs = int(original_fs * l_factor / m_factor)
    
    try:
        # resample_poly es la forma correcta ingenierilmente: aplica filtro FIR
        # para evitar aliasing (al diezmar) o imágenes (al interpolar)
        resampled_data = signal.resample_poly(data, l_factor, m_factor)
    except Exception as e:
        # Si falla (ej. memoria), devolvemos original
        print(f"Error en resampling: {e}")
        return data, original_fs
    
    return resampled_data, new_fs

def design_safe_filter(lowcut, highcut, fs, order=2):
    """
    Diseña un filtro solo si las frecuencias son válidas para la Fs actual.
    Retorna filtros SOS (más estables).
    """
    nyquist = 0.5 * fs
    
    # CASO CRÍTICO: El filtro está totalmente fuera del rango auditivo actual
    if lowcut >= nyquist:
        return None 

    # CASO LÍMITE: La frecuencia alta toca o excede Nyquist
    # La recortamos al 95% de Nyquist para evitar inestabilidad matemática
    if highcut >= nyquist:
        highcut = nyquist * 0.95
    
    # Evitar que low y high se crucen tras el ajuste
    if lowcut >= highcut:
        return None

    try:
        # Normalizar respecto a Nyquist
        sos = signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
        return sos
    except Exception:
        return None

def apply_equalizer(data, fs, gains_db):
    """
    Aplica ecualizador gráfico de 6 bandas con suma ponderada.
    """
    bands = {
        "Sub-Bass": (16, 60),
        "Bass": (60, 250),
        "Low Mids": (250, 2000),
        "High Mids": (2000, 4000),
        "Presence": (4000, 6000),
        "Brilliance": (6000, 16000)
    }
    
    # Creamos un acumulador de ceros del mismo tamaño que la señal
    output_signal = np.zeros_like(data)
    
    # Variable para saber si al menos un filtro funcionó
    active_filters = False

    for band_name, (low, high) in bands.items():
        # Intentar diseñar filtro seguro
        sos = design_safe_filter(low, high, fs)
        
        if sos is not None:
            active_filters = True
            # Filtrar la señal original para extraer esta banda
            try:
                band_component = signal.sosfilt(sos, data)
                
                # Aplicar ganancia logarítmica
                gain_val = gains_db.get(band_name, 0)
                gain_linear = 10 ** (gain_val / 20.0)
                
                # Sumar al mix final
                output_signal += band_component * gain_linear
            except:
                pass # Si explota un filtro, lo ignoramos

    # Si ningún filtro fue válido (ej. Fs muy baja), pasamos la señal original (Bypass)
    # o devolvemos silencio si se prefiere. Aquí devolvemos señal atenuada para evitar silencio total.
    if not active_filters:
        return data

    # SANITIZACIÓN FINAL DEL MOTOR DSP
    # Convertir NaNs (Not a Number) e Infs a 0.0
    output_signal = np.nan_to_num(output_signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    return output_signal

def compute_fft(data, fs):
    n = len(data)
    if n == 0: return [0], [0]
    
    # Ventana de Hanning para suavizar bordes espectrales
    window = np.hanning(n)
    
    # FFT Real
    try:
        freq = np.fft.rfftfreq(n, d=1/fs)
        magnitude = np.abs(np.fft.rfft(data * window))
    except:
        return [0], [0]
        
    return freq, magnitude
