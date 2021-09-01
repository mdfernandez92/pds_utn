import numpy as np

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    """Generador de señales senoidales parametrizable
    
    Es importante que fs >= 2*ff por teorema de muestreo.

    Args:
        vmax (float):   amplitud máxima de la senoidal (volts)
        dc (float):     valor medio (volts)
        ff (float):     frecuencia (Hz)
        ph (float):     fase (radianes)
        nn (int):       cantidad de muestras digitalizada por el ADC (# muestras)
        fs (float):     frecuencia de muestreo del ADC

    Returns:
        tt (float[nn]): Base de tiempos de la señal generada
        xx (float[nn]): Valores de amplitud de la señal generada
    """
    # Calculamos el tiempo de muestreo
    ts = 1/fs
    
    # Calculamos la frecuencia angular
    w0 = 2*np.pi*ff
    
    # Generamos la base de tiempos
    tt = np.linspace(0, (nn-1)*ts, num=nn).flatten()
    
    # Generamos la amplitud de la señal para cada instante en la base de tiempos
    xx = vmax * np.sin(w0*tt+ph) + dc

    # Retornamos espacio de tiempos y señal
    return tt, xx

def DFT(x,fs):
    nn = x.shape[0]
    # Generamos un array con ceros para guardar las amplitudes en frecuencia
    X = np.array([0+0j]).repeat(nn)
    
    tt = np.linspace(0, (nn-1), num=nn)
    tt *= 1/(nn*(1/fs))
    
    for k in range(0,nn-1):
        for n in range(0,nn-1):
            X[k] += x[n] * np.exp((-1j)*2*np.pi*(k*n)/nn)

    return tt, X


import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuración del tamaño de los gráficos
mpl.rcParams['figure.figsize'] = (8,5)

# Configuración del tamaño de fuente
fig_font_size = 14
plt.rcParams.update({'font.size':fig_font_size})


# Señal 1: Amplitud = 1V, Valor Medio = 0V, Frec = 10Hz, Fase = 0 rad, N° muestras ADC = 100*10, Frec. muestreo ADC = 1000Hz
tt, xx = mi_funcion_sen(vmax = 1, dc = 0, ff = 15, ph=0, nn = 100*10, fs = 1000)

kk, X = DFT(xx,fs=1000)

# Graficación de señal
plt.figure(1)
plt.plot(tt, xx)
plt.title('Señal senoidal 1 - 10Hz - 10 ciclos')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.show()

# Graficación de señal
plt.figure(2)
plt.bar(kk, np.abs(X))
plt.xlim([0,20])
plt.title('DFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('X[k]')
plt.show()