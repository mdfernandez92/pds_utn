import numpy as np
from numpy import random

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

def mi_funcion_square(vmax, dc, ff, ph, nn, fs):
    """Generador de señales cuadradas parametrizable
    
    Es importante que fs >= 2*ff por teorema de muestreo.
    El duty es fijo en 50%
    
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
    # Utilizamos señal senoidal como la forma base
    tt, xx = mi_funcion_sen(1, 0, ff, ph, nn, fs)
    
    xx[xx>=0] =  vmax + dc
    xx[xx<0]  = -vmax + dc
    
    # Retornamos espacio de tiempos y señal
    return tt, xx

def mi_funcion_uniform(var,nn,fs,a=0):
    """Generador de señal aleatoria uniforme
    
    Args:
        var (float):   varianza de la señal a generar
        nn  (int):     cantidad de muestras de la señal
        fs  (float):   frecuencia de muestreo
        a   (float):   valor inicial del rango aleatorio. default = 0 

    Returns:
        xx (float[nn]): Valores de amplitud de la señal generada
    """
    # Calculamos el tiempo de muestreo
    ts = 1/fs
    
    # Generamos la base de tiempos
    tt = np.linspace(0, (nn-1)*ts, num=nn).flatten()
    
    # Los valores aleatorios son de 0 en adelante
    a = 0
    
    # Calculamos (b - a) a partir de la varianza
    b_a = (12/2)*np.sqrt(var)
    
    # Generamos señal aleatoria
    xx = b_a * random.random_sample(nn) + a

    return tt, xx
    
    