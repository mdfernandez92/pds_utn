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
    b_a = np.sqrt(12*var)
    
    # Generamos señal aleatoria
    xx = b_a * random.random_sample(nn) + a

    return tt, xx

def mi_cuantizador(sr,B,VF):
    """Cuantiza una señal discreta recibida como parámetro utilizando B bits en el rango VF
    
    Args:
        sr (float[nn]): señal a cuantizar, una matriz (Nx1) de números reales. 
        B  (int):       cantidad de Bits a utilizar para la cuantización
        VF (float):     valor absoluto del máximo valor del rango de amplitudes de la señal a cuantizar

    Returns:
        sq (float[nn]): señal cuantizada
    """
    # Calculamos paso de cuantización
    q = VF / (2**B)
    
    # Cuantizamos la señal recibida como parámetro
    sq = q * np.round(sr/q)
        
    return sq

def window_rectangular(L):
    """ Implementación de ventana rectangular
    
    Args:
        L longitud de la ventana en cantidad de muestras
        
    Returns:
        w amplitudes de la ventana en cada muestra
    """
    
    # Validación de la cantidad de muestras
    if L == 1:
        return 1
    
    # Valor máximo de indice
    N = L - 1
    
    # Generamos grilla para almacenar amplitudes de la ventana para cada muestra
    # Temporalmente el valor de cada posición es la del indice a fin de usar como referencia
    w = np.linspace(0,N,L)

    # Asignamos valor unitario a todas las posiciones del vector
    w[:]  = 1
    
    return w

def window_bartlett(L):
    """ Implementación de ventana Bartlett
    
    Args:
        L longitud de la ventana en cantidad de muestras
        
    Returns:
        w amplitudes de la ventana en cada muestra
    """
    
    # Validación de la cantidad de muestras
    if L == 1:
        return 1

    # Valor máximo de indice
    N = L - 1
    
    # Generamos grilla para almacenar amplitudes de la ventana para cada muestra
    # Temporalmente el valor de cada posición es la del indice a fin de usar como referencia
    w = np.linspace(0,N,L)
    
    # Generamos indexadores booleanos acorde expresión analítica de ventana
    b1 = ((w >= 0)   & (w <= N/2))
    b2 = ((w >= N/2) & (w <= N))
    
    # Asignamos valores de amplitud de la ventana acorde indexadores booleanos generados
    w[b1] = 2 * w[b1] / N 
    w[b2] = 2 - (2 * w[b2] / N)
    
    return w

def window_hann(L):
    """ Implementación de ventana Hann
    
    Args:
        L cantidad de muestras
        
    Returns:
        w amplitudes de la ventana en cada muestra
    """
    
    # Validación de la cantidad de muestras
    if L == 1:
        return 1

    # Valor máximo de indice
    N = L - 1

    # Generamos grilla para almacenar amplitudes de la ventana para cada muestra
    # Temporalmente el valor de cada posición es la del indice a fin de usar como referencia
    w = np.linspace(0,N,L)
    
    # Generamos indexador booleano acorde expresión analítica de ventana
    b1 = ((w >= 0)   & (w <= N))
    
    # Asignamos valores de amplitud de la ventana acorde indexador booleano generado
    w[b1] = 0.5 - 0.5 * np.cos(2*np.pi*w[b1]/N)
    
    return w    

def window_blackman(L):
    """ Implementación de ventana Blackman
    
    Args:
        L cantidad de muestras
        
    Returns:
        w amplitudes de la ventana en cada muestra
    """
    
    # Validación de la cantidad de muestras
    if L == 1:
        return 1

    # Valor máximo de indice
    N = L - 1

    # Generamos grilla para almacenar amplitudes de la ventana para cada muestra
    # Temporalmente el valor de cada posición es la del indice a fin de usar como referencia
    w = np.linspace(0,N,L)
    
    # Generamos indexador booleano acorde expresión analítica de ventana
    b1 = ((w >= 0)   & (w <= N))
    
    # Asignamos valores de amplitud de la ventana acorde indexador booleano generado
    w[b1] = 0.42 - 0.5 * np.cos(2*np.pi*w[b1]/N) + 0.08 * np.cos(4*np.pi*w[b1]/N) 
    
    return w  

def window_flat_top(L):
    """ Implementación de ventana Blackman
    
    Args:
        L cantidad de muestras
        
    Returns:
        w amplitudes de la ventana en cada muestra
    """
    
    # Validación de la cantidad de muestras
    if L == 1:
        return 1
    
    # Valor máximo de indice
    N = L - 1
    
    # Definimos amplitud de los cosenos que forman la ventana
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368

    # Generamos grilla para almacenar amplitudes de la ventana para cada muestra
    # Temporalmente el valor de cada posición es la del indice a fin de usar como referencia
    w = np.linspace(0,N,L)
    
    # Generamos indexador booleano acorde expresión analítica de ventana
    b1 = ((w >= 0)   & (w <= N))
    
    # Asignamos valores de amplitud de la ventana acorde indexador booleano generado
    w[b1] = a0 - a1 * np.cos(2*np.pi*w[b1]/N) + a2 * np.cos(4*np.pi*w[b1]/N) - a3 * np.cos(6*np.pi*w[b1]/N) + a4 * np.cos(8*np.pi*w[b1]/N)  
        
    return w  