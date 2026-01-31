import matplotlib.pyplot as plt

# ... (resto de tu código anterior) ...

# --- VISUALIZACIÓN ---
st.divider()

# Selector de Estilo
viz_style = st.radio("Estilo de Visualización:", ["Ingeniería (Rápido)", "Teórico (Oppenheim Style)"], horizontal=True)

if viz_style == "Ingeniería (Rápido)":
    # --- MODO INTERACTIVO (EL QUE YA TENÍAS) ---
    col_viz, col_play = st.columns([3, 1])
    
    with col_viz:
        tab1, tab2 = st.tabs(["📈 Tiempo (Envolvente)", "🌊 Frecuencia (Analizador)"])
        
        # ... (Aquí va tu código de Plotly anterior para fig_t y fig_f) ...
        # (Si necesitas que te repita este bloque dímelo, pero asumo que ya lo tienes)
        # Usa las gráficas go.Scatter con líneas que hicimos antes.
        
        # MANTÉN TU CÓDIGO DE PLOTLY AQUÍ PARA EL MODO RÁPIDO

else:
    # --- MODO TEÓRICO (TIPO LIBRO OPPENHEIM / GUÍA) ---
    st.markdown("### 🔬 Visualización de Muestras Discretas (Zoom)")
    st.caption("Visualizando solo 50 muestras para apreciar el efecto de muestreo $x[n]$ vs $y[n]$.")
    
    # Tomamos un slice muy pequeño (Microscopio)
    n_samples = 50
    start_idx = len(work_data) // 2 # Justo al medio
    
    # Datos Recortados
    slice_in = normalize_visuals(work_data[start_idx : start_idx + n_samples])
    
    # Para la salida, calculamos el índice equivalente según el cambio de tasa
    ratio = new_fs / fs_in
    start_idx_out = int(start_idx * ratio)
    n_samples_out = int(n_samples * ratio)
    
    slice_out = normalize_visuals(processed[start_idx_out : start_idx_out + n_samples_out])

    # --- GRAFICAR CON MATPLOTLIB (ESTILO STEM) ---
    fig_opp, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    
    # Configuración estética tipo Libro
    marker_style = dict(markerfacecolor='black', markeredgecolor='black', markersize=4)
    stem_style = dict(linefmt='k-', basefmt='k-', use_line_collection=True)
    
    # Gráfica 1: Entrada x[n]
    ax1.set_title(r"Señal Original $x[n]$ (Tiempo Discreto)", fontsize=12)
    ax1.stem(range(len(slice_in)), slice_in, **stem_style, markerfmt='ko')
    ax1.set_ylabel("Amplitud")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Gráfica 2: Salida y[n]
    ax2.set_title(r"Señal Procesada $y[n]$ (Diezmada/Interpolada)", fontsize=12)
    # Ajustamos el eje X de la salida para reflejar el cambio de densidad
    x_axis_out = np.linspace(0, len(slice_in), len(slice_out))
    
    # Usamos color rojo para diferenciar, pero manteniendo estilo stem
    ax2.stem(x_axis_out, slice_out, linefmt='r-', basefmt='k-', markerfmt='ro', use_line_collection=True)
    ax2.set_xlabel("Muestras ($n$)")
    ax2.set_ylabel("Amplitud")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    st.pyplot(fig_opp)
    
    # --- ESPECTRO TEÓRICO (ÁNGULO) ---
    st.markdown("### 📐 Espectro en Frecuencia Angular ($\omega$)")
    
    # FFT de alta resolución para ver curvas suaves
    N_fft = 2048
    W_in = np.fft.fftshift(np.fft.fft(work_data[:N_fft]))
    W_out = np.fft.fftshift(np.fft.fft(processed[:int(N_fft*ratio)]))
    
    freqs_in = np.linspace(-np.pi, np.pi, len(W_in))
    freqs_out = np.linspace(-np.pi, np.pi, len(W_out))
    
    mag_in = 20 * np.log10(np.abs(W_in) + 1e-9)
    mag_out = 20 * np.log10(np.abs(W_out) + 1e-9)

    fig_w, ax3 = plt.subplots(figsize=(10, 4))
    
    ax3.plot(freqs_in, mag_in, 'k-', alpha=0.3, label=r'Original $X(e^{j\omega})$')
    ax3.plot(freqs_out, mag_out, 'r-', linewidth=1.5, label=r'Procesada $Y(e^{j\omega})$')
    
    ax3.set_title(r"Espectro de Frecuencia Normalizada ($-\pi$ a $\pi$)")
    ax3.set_xlabel(r"Frecuencia Angular $\omega$ (rad)")
    ax3.set_ylabel("Magnitud (dB)")
    ax3.set_xlim([-np.pi, np.pi])
    
    # Ticks en pi
    ax3.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax3.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    st.pyplot(fig_w)

# ... (El resto de la columna de audio 'col_play' sigue igual) ...
