# Importamos módulos a utilizar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import EngFormatter

plt.close("all")

# Configuramos tamaño de gráficos
mpl.rcParams['figure.figsize'] = (7,5)

# Configuramos tamaño de fuentes de gráficos
plt.rcParams.update({'font.size': 10})

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


# Definimos la cantidad de muestras para todas las ventanas
N = 1000

# Obtenemos las amplitudes para cada ventana
ww_rectangular = window_rectangular(N)
ww_bartlett    = window_bartlett(N)
ww_hann        = window_hann(N)
ww_blackman    = window_blackman(N)
ww_flat_top    = window_flat_top(N)

# Graficamos las ventanas en el dominio del tiempo
plt.figure()
plt.title("Ventanas en el dominio temporal - $w(k)$")
plt.plot(ww_rectangular, label="Rectangular")
plt.plot(ww_bartlett,    label="Bartlett")
plt.plot(ww_hann,        label="Hann")
plt.plot(ww_blackman,    label="Blackman")
plt.plot(ww_flat_top,    label="Flat-top")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
plt.show()



# Definimos la cantidad de padding a aplicar
cant_pad = 9

# Generamos un vector de ceros con el tamaño del vector original a paddear
zz = np.zeros(N)

# Repetimos el vector tantas veces como padding a aplicar
padding = zz.repeat(cant_pad, axis = 0)

# Aplicamos el padding a las ventanas generadas
ww_rectangular_pad = np.hstack([ww_rectangular, padding])
ww_bartlett_pad    = np.hstack([ww_bartlett,    padding])
ww_hann_pad        = np.hstack([ww_hann,        padding])
ww_blackman_pad    = np.hstack([ww_blackman,    padding])
ww_flat_top_pad    = np.hstack([ww_flat_top,    padding])

# Normalizamos las ventanas por el valor medio
ww_rectangular_pad = ww_rectangular_pad / np.mean(ww_rectangular_pad)
ww_bartlett_pad    = ww_bartlett_pad    / np.mean(ww_bartlett_pad)
ww_hann_pad        = ww_hann_pad        / np.mean(ww_hann_pad)
ww_blackman_pad    = ww_blackman_pad    / np.mean(ww_blackman_pad)
ww_flat_top_pad    = ww_flat_top_pad    / np.mean(ww_flat_top_pad)

# Obtenemos el nuevo tamaño de los vectores con padding
N_Pad = ww_bartlett_pad.shape[0]

# Definimos frecuencia de muestreo
# Nota: Utilizamos una fs distinta de 1000 Hz para que quede en evidenciada la normalización de la frecuencia
fs = 2000

# Calculamos resolución espectral de la señal original
df = fs/N

# Calculamos resolución espectral para señales con padding
df_p = fs/N_Pad

# Generamos grilla frecuencial para señales con padding
ff = np.linspace(0, (N_Pad-1)*df_p, N_Pad)

# Definimos la norma para normalización de la frecuencia
# Queremos la frecuencia normalizada sea en función de la resolución espectral original sin padding. 
# 1 = df, 2 = 2df, 3 = 3df, etc.
norm = 1/df

# Normalizamos la grilla frecuencial
ff *= norm

# Generamos indexador de los bines acorde rango de frecuencias normalizadas de interés
bfrec = ff <= ((fs/2) * norm)
    
# Calculamos la FFT de las ventanas generadas
WW_rectangular_pad = 1/N_Pad * np.fft.fft(ww_rectangular_pad)
WW_bartlett_pad    = 1/N_Pad * np.fft.fft(ww_bartlett_pad)
WW_hann_pad        = 1/N_Pad * np.fft.fft(ww_hann_pad)
WW_blackman_pad    = 1/N_Pad * np.fft.fft(ww_blackman_pad)
WW_flat_top_pad    = 1/N_Pad * np.fft.fft(ww_flat_top_pad)

# Para evitar aplicar logaritmo a amplitudes iguales a 0, vamos a sumar resolución de flotante a cada amplitud
eps = np.finfo(float).eps

# Espectros de amplitud en dB de cada ventana
WW_rectangular_db = 20*np.log10(np.abs(WW_rectangular_pad[bfrec])+eps)
WW_bartlett_db    = 20*np.log10(np.abs(WW_bartlett_pad[bfrec])+eps)
WW_hann_db        = 20*np.log10(np.abs(WW_hann_pad[bfrec])+eps)
WW_blackman_db    = 20*np.log10(np.abs(WW_blackman_pad[bfrec])+eps)
WW_flat_top_db    = 20*np.log10(np.abs(WW_flat_top_pad[bfrec])+eps)

# Graficamos los espectros
fig, ax = plt.subplots()
plt.title("Espectros de amplitud de las ventanas en el dominio frecuencial - $|W(Ω)|$")
ax.plot(ff[bfrec],WW_rectangular_db,':x', label="Rectangular")
ax.plot(ff[bfrec],WW_bartlett_db,   ':x', label="Bartlett")
ax.plot(ff[bfrec],WW_hann_db,       ':x', label="Hann")
ax.plot(ff[bfrec],WW_blackman_db,   ':x', label="Blackman")
ax.plot(ff[bfrec],WW_flat_top_db,   ':x', label="Flat-top")
ax.set_xlim([0,10])
ax.set_ylim([-100,0])
ax.xaxis.set_major_formatter(EngFormatter(unit=u"$\Delta_f$"))
ax.set_ylabel("|W(Ω)| [dB]")
ax.set_xlabel("Frecuencia Normalizada: $Ω = f $ / $\Delta_f$")
ax.set_xticks(np.linspace(0, (11-1), 11))
ax.legend()
plt.show()



from pandas import DataFrame
from IPython.display import HTML

# Indices del primer cruce por 0 
# Lo buscamos como aquel donde la amplitud en db se encuentra por debajo de un umbral en dB
idx_rectangular_first_0 = np.where(WW_rectangular_db <= -100)[0][0]
idx_bartlett_first_0    = np.where(WW_bartlett_db    <= -100)[0][0]
idx_hann_first_0        = np.where(WW_hann_db        <= -60)[0][0]
idx_blackman_first_0    = np.where(WW_blackman_db    <= -100)[0][0]
idx_flat_top_first_0    = np.where(WW_flat_top_db    <= -100)[0][0]

# Obtenemos frecuencias asociadas
rectangular_first_0 = ff[idx_rectangular_first_0]
bartlett_first_0    = ff[idx_bartlett_first_0]
hann_first_0        = ff[idx_hann_first_0]
blackman_first_0    = ff[idx_blackman_first_0]
flat_top_first_0    = ff[idx_flat_top_first_0]

# Indices -3dB
# Lo buscamos como aquel donde la amplitud en db se encuentra por debajo de un umbral en dB
rectangular_3db = ff[np.where(WW_rectangular_db <= -3)[0][0]]
bartlett_3db    = ff[np.where(WW_bartlett_db    <= -3)[0][0]]
hann_3db        = ff[np.where(WW_hann_db        <= -3)[0][0]]
blackman_3db    = ff[np.where(WW_blackman_db    <= -3)[0][0]]
flat_top_3db    = ff[np.where(WW_flat_top_db    <= -3)[0][0]]

rectangular_w2 = np.max(WW_rectangular_db[idx_rectangular_first_0:])
bartlett_w2    = np.max(WW_bartlett_db   [idx_bartlett_first_0   :])
hann_w2        = np.max(WW_hann_db       [idx_hann_first_0       :])
blackman_w2    = np.max(WW_blackman_db   [idx_blackman_first_0   :])
flat_top_w2    = np.max(WW_flat_top_db   [idx_flat_top_first_0   :])

data = [["${}\ \Delta_f$".format(np.round(rectangular_first_0,2)), "${}\ \Delta_f$".format(np.round(rectangular_3db,2)), "${}\ dB$".format(np.round(rectangular_w2))],
        ["${}\ \Delta_f$".format(np.round(bartlett_first_0,2)),    "${}\ \Delta_f$".format(np.round(bartlett_3db,2)),    "${}\ dB$".format(np.round(bartlett_w2   ))],
        ["${}\ \Delta_f$".format(np.round(hann_first_0,2)),        "${}\ \Delta_f$".format(np.round(hann_3db,2)),        "${}\ dB$".format(np.round(hann_w2       ))],
        ["${}\ \Delta_f$".format(np.round(blackman_first_0,2)),    "${}\ \Delta_f$".format(np.round(blackman_3db,2)),    "${}\ dB$".format(np.round(blackman_w2   ))],
        ["${}\ \Delta_f$".format(np.round(flat_top_first_0,2)),    "${}\ \Delta_f$".format(np.round(flat_top_3db,2)),    "${}\ dB$".format(np.round(flat_top_w2   ))]
       ]

df = DataFrame(data,columns=['$\Omega_0$', '$\Omega_1$', '$W_2$' ],
                index=[  
                        'Rectangular',
                        'Bartlett',
                        'Hann',
                        'Blackman',
                        'Flat-top'])
HTML(df.to_html())