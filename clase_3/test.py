# -*- coding: utf-8 -*-

# Importamos desde biblioteca PDS generadores de señales realizados previamente 
from pds import mi_funcion_sen, mi_funcion_square

# Importamos biblioteca para gráficos
import matplotlib.pyplot as plt
import matplotlib as mpl

# Importamos biblioteca numpy
import numpy as np

# Importamos biblioteca para cálculo de fft
from scipy.fft import fft

# Configuración del tamaño de fuente
fig_font_size = 14
plt.rcParams.update({'font.size':fig_font_size})

# Configuración del tamaño de los gráficos
mpl.rcParams['figure.figsize'] = (8,5)

# Definimos parámetros de la señal senoidal
vmax = 4     # Amplitud [V]
dc   = 0     # Valor Medio [V]
f0   = 1     # Frecuencia [Hz]
ph   = 0     # Fase [rad]
N    = 1000  # N° muestras ADC
fs   = 1000  # Frecuencia de muestreo ADC [Hz] 

# Generamos señal utilizando los parámetros definidos
tt, xx = mi_funcion_sen(vmax = vmax, dc = dc, ff = f0, ph = ph, nn = N, fs = fs)
    
# # Graficación de señal en el dominio del tiempo
# plt.plot(tt, xx)
# plt.title('Señal senoidal - 1Hz - 1 ciclo')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud [V]')
# plt.show()

# Calculamos las FFT
XX_fft = fft(xx) / N

# # Calculamos el paso frecuencial
# df = 1/(N*(1/fs))

# # Generamos base de frecuencias para graficar DFT
# ff = np.linspace(0, (N-1), num=N) * df

# Configuración del tamaño de los gráficos
# mpl.rcParams['figure.figsize'] = (15,5)

# # Graficación de señal en el dominio de la frecuencia
# figure, axes = plt.subplots(nrows=1, ncols=2)

# axes[0].stem(ff, np.abs(XX_fft))
# axes[0].set_title('FFT - Espectro de amplitud')
# axes[0].set_xlim([0,10])
# axes[0].set_xlabel('Frecuencia [Hz]')
# axes[0].set_ylabel('|X(f)|')

# axes[1].stem(ff, np.angle(XX_fft))
# axes[1].set_title('FFT - Diagrama de fases')
# axes[1].set_xlim([0,10])
# axes[1].set_xlabel('Frecuencia [Hz]')
# axes[1].set_ylabel('angle(X(f))')

# plt.show()

# figure.tight_layout()

energia_tiempo = sum(xx**2) / N
energia_freq   = sum(np.abs(XX_fft)**2)
varianza       = np.var(xx)