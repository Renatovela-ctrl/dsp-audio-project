# 🎚️ EcoLab: Audio Loop & DSP Station

**EcoLab** es una aplicación web interactiva para el registro, visualización y procesamiento de señales de audio en tiempo real. Este proyecto combina tecnologías web modernas con un motor de procesamiento digital de señales (DSP) robusto en Python.

Desarrollado como parte de prácticas de ingeniería en la **Universidad de Cuenca**.

## 🚀 Características Principales

### 🎨 Frontend (Visualización)
- **Osciloscopio en Tiempo Real:** Visualización de la forma de onda (dominio del tiempo) utilizando la API de Canvas.
- **Analizador de Espectro:** Visualización de frecuencias (FFT) con gradientes de color reactivos.
- **Interfaz Intuitiva:** Diseño "Dark Mode" estilo rack de estudio con controles ergonómicos.
- **Loop Station:** Funcionalidad de bucle (toggleable) para repetición de muestras (default 15s).

### 🧮 Backend (Motor DSP)
El núcleo del procesamiento de audio (`dsp.py`) implementa técnicas avanzadas de ingeniería:
- **Filtros SOS (Second-Order Sections):** Uso de filtros `butter` en configuración SOS para máxima estabilidad numérica.
- **Protección de Nyquist:** Algoritmos de seguridad que ajustan o desactivan filtros automáticamente si las frecuencias de corte se acercan a $F_s/2$ para evitar aliasing o inestabilidad.
- **Resampling Polifásico:** Implementación de `signal.resample_poly` para cambios de tasa de muestreo con filtrado antialiasing integrado.
- **Ecualizador Paramétrico:** Procesamiento por bandas separadas con suma ponderada.

## 🛠️ Tecnologías Utilizadas

* **Python 3.x**: Lenguaje principal.
* **Flask**: Micro-framework para el servidor web.
* **NumPy & SciPy**: Librerías para cálculo matemático y procesamiento de señales.
* **SoundFile**: Lectura y escritura de buffers de audio.
* **HTML5 / CSS3 / JavaScript**: Interfaz de usuario y API de Web Audio.

## 📦 Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/ecolab-audio.git](https://github.com/tu-usuario/ecolab-audio.git)
    cd ecolab-audio
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    # En Windows:
    venv\Scripts\activate
    # En Mac/Linux:
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install flask numpy scipy soundfile
    ```

4.  **Ejecutar la aplicación:**
    ```bash
    python app.py
    ```

5.  **Abrir en el navegador:**
    Ve a `http://127.0.0.1:5000/` y permite el acceso al micrófono.

## 📂 Estructura del Proyecto

```text
ecolab-audio/
├── app.py              # Servidor Flask (Controlador)
├── dsp.py              # Módulo de Procesamiento Digital de Señales (Lógica)
├── templates/
│   └── index.html      # Interfaz de Usuario (Visualizadores JS)
├── static/             # Archivos estáticos (si aplica)
└── README.md           # Documentación
