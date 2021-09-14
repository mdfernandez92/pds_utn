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
vmax = 1     # Amplitud [V]
dc   = 0     # Valor Medio [V]
f0   = 1     # Frecuencia [Hz]
ph   = np.pi/4     # Fase [rad]
N    = 1000  # N° muestras ADC
fs   = 1000  # Frecuencia de muestreo ADC [Hz] 

# Generamos señal utilizando los parámetros definidos
tt, xx_1 = mi_funcion_sen(vmax = vmax, dc = dc, ff = f0, ph = ph, nn = N, fs = fs)
    
# Generamos señal utilizando los parámetros definidos
tt, xx_2 = mi_funcion_sen(vmax = vmax, dc = dc, ff = f0, ph = ph, nn = N, fs = fs)

xx_3 = np.dot(xx_1,xx_2)
 
# Graficación de señal en el dominio del tiempo
plt.plot(tt, xx_3)
plt.title('Señal senoidal - 1Hz - 1 ciclo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.show()
