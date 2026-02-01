# Laboratorio de Procesamiento de Señales en Tiempo Discreto 🎛️

### Autores:
* **Israel Méndez**
* **Daniel Molina**
* **Renato Vela**

**Institución:** Universidad de Cuenca  
**Asignatura:** Sistemas Lineales y Señales  
**Tema:** T3 - Procesamiento en TD de Señales en TC

---

## 📝 Descripción del Proyecto

Este sistema integral de **Procesamiento de Señales en Tiempo Discreto (DSP)** permite la manipulación y análisis de señales de audio $x[n]$ mediante algoritmos manuales. El proyecto se centra en la aplicación práctica de la teoría de **Sistemas Lineales e Invariantes en el Tiempo (LTI)**, cubriendo el muestreo, la conversión de tasa y el filtrado digital.



## 🚀 Características Técnicas

### 1. Conversor de Tasa de Muestreo (SRC)
Implementación de un sistema polifásico para la modificación de la frecuencia de muestreo original $F_s$ mediante los factores $L$ (Expansión) y $M$ (Diezmado):
* **Expansión ($L$):** Inserción de ceros entre muestras para aumentar la tasa de muestreo.
* **Filtro de Interpolación:** Filtro Paso Bajo (LPF) diseñado con núcleo Sinc y ventana Blackman para eliminar imágenes espectrales.
* **Diezmado ($M$):** Reducción de la tasa de muestreo mediante filtrado anti-solapamiento (Anti-aliasing) previo para cumplir con el Teorema de Nyquist.

### 2. Ecualizador Multibanda
Banco de filtros IIR de segundo orden (Biquad) diseñados mediante la **Transformada Bilineal**. El sistema permite el control de magnitud en las siguientes bandas:
* **Sub-Bass:** 16-60 Hz
* **Bass:** 60-250 Hz
* **Low Mids:** 250-2000 Hz
* **High Mids:** 2000-4000 Hz
* **Presence:** 4000-6000 Hz
* **Brilliance:** 6000-16000 Hz

### 3. Análisis Espectral (FFT Manual)
Implementación propia del algoritmo de **Transformada Rápida de Fourier (FFT)** por diezmado en el tiempo (Radix-2) para visualizar el espectro de magnitud $|X(e^{j\omega})|$.



## 📐 Fundamentos Teóricos Aplicados

* **Teorema de Nyquist:** Ajuste dinámico de filtros para prevenir aliasing cuando $F_s$ disminuye.
* **Frecuencia Angular Normalizada:** Visualización opcional en $rad/s$, donde $\pi$ representa la frecuencia de Nyquist.
* **Simetría Conjugada:** Aprovechamiento de la propiedad de paridad en señales reales para el análisis espectral.

## 📦 Requisitos e Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/nombre-del-repo.git](https://github.com/tu-usuario/nombre-del-repo.git)
    ```
2.  **Instalar dependencias:**
    ```bash
    pip install streamlit numpy plotly soundfile scipy matplotlib
    ```
3.  **Ejecutar la aplicación:**
    ```bash
    streamlit run app.py
    ```

## 📋 Estructura de Archivos

* `app.py`: Interfaz gráfica desarrollada en Streamlit.
* `modules/dsp_core.py`: Núcleo con algoritmos manuales de FFT, convolución y ecuaciones en diferencias.
* `examples/`: Directorio de archivos `.wav` para pruebas.

---
**Nota Académica:** Este proyecto evita el uso de funciones de alto nivel para el procesamiento (como `resample` o `filtfilt`), optando por implementaciones manuales que demuestran la comprensión de la teoría de Señales y Sistemas.
