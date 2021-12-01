# Importamos bibliotecas a utilizar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as sig

plt.close("all")

# Configuramos tamaño de gráficos
mpl.rcParams['figure.figsize'] = (14,7)

# Configuramos tamaño de fuentes de gráficos
plt.rcParams.update({'font.size': 16})

# Para evitar aplicar logaritmo a amplitudes iguales a 0, vamos a sumar resolución de flotante a cada amplitud
eps = np.finfo(float).eps

# Definimos parámetros de la señal senoidal a generar
N  = 1000 # Cantidad de muestras
fs = 1000 # Frecuencia de muestreo

# Calculamos el tiempo de muestreo
ts = 1/fs

# Generamos la base de tiempos
tt = np.linspace(0, (N-1)*ts, num=N).reshape(N,1)

# Definimos cantidad de realizaciones
R = 200

# Obtenemos las realizaciones de la variable aleatoria
fr = np.random.uniform(-1/2,1/2,R)

# Calculamos la frecuencia angular de la señal
Ω0 = np.pi / 2
Ω1 = Ω0 + fr * ((2*np.pi)/N)

# Definimos parámetro normalizador
norm = (fs / (2*np.pi))

# Desnormalizamos la frecuencia angular
f0 = Ω1 * norm

# Transformamos las dimensiones del array para poder multiplicar por tt
f0 = f0.reshape(1,R)

# Definimos la potencia de la señal senoidal como unitaria
pot_sen = 1

# Calculamos la potencia de la señal senoidal en dB
pot_sen_db = 10*np.log10(pot_sen)

# Calculamos la amplitud de la señal senoidal a partir de la potencia (en veces)
a1 = np.sqrt(2*pot_sen)

# Generamos la señal senoidal
xx_sen = a1 * np.sin(2*np.pi * f0 * tt)

# Definimos SNR
SNRs = np.array([3,10])

for snr in SNRs:
    # Calculamos la potencia del ruido en dB
    pot_ruido_db = pot_sen_db - snr

    # Calculamos la potencia del ruido en veces
    pot_ruido = 10**(pot_ruido_db/10)

    # Definimos valor medio y desvío estandard del ruido
    # El desvío standard es la raiz de la varianza del ruido y la varianza es la potencia del mismo
    noise_mean = 0
    noise_var  = pot_ruido
    noise_std  = np.sqrt(noise_var)

    # Generamos la señal de ruido
    noise = np.random.normal(noise_mean,noise_std,N)

    # Transformamos las dimensiones del array
    noise = noise.reshape(N,1)
    
    # Generamos la señal final a partir de la senoidal y el ruido
    xx = xx_sen + noise
    
    # Definimos padding a aplicar a la señal
    N_pad = N * 10
    
    # Calculamos el PSD mediante método periodograma
    xx_fft = 1/N_pad * np.fft.fft(xx, n=N_pad, axis=0)
    psd_period = np.abs(xx_fft)**2

    # Calculamos el PSD mediante método de Welch
    f, psd_welch = sig.welch(xx, fs=fs, nfft=N_pad, axis=0)
    
    # Graficamos PSD Periodograma
    plt.figure()
    plt.plot(f,psd_period[bfrec])
    plt.title("PSD - Periodograma para SNR={:d}dB".format(snr))
    plt.ylabel("PSD")
    plt.xlabel("Frecuencia (Hz)")
    plt.xlim(240,260)
    plt.show()
    
    # Graficamos PSD Welch
    plt.figure()
    plt.plot(f,psd_welch)
    plt.title("PSD - Welch para SNR={:d}dB".format(snr))
    plt.ylabel("PSD")
    plt.xlabel("Frecuencia (Hz)")
    plt.xlim(240,260)
    plt.show()

