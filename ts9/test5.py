# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 01:47:14 2021

@author: Mauro
"""

# %% Inicialización y alineación de latidos

# Importamos bibliotecas a utilizar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio

# Cerramos todos los gráficos
plt.close("all")

# Configuramos tamaño de gráficos
mpl.rcParams['figure.figsize'] = (10,5)

# Configuramos tamaño de fuentes de gráficos
plt.rcParams.update({'font.size': 12})

# Leemos el dataset desde el archivo ECG.mat
mat_struct = sio.loadmat("ecg.mat")

# Serie temporal con el registro del ECG
xx = mat_struct["ecg_lead"]

# Índices con la ubicación de los latidos detectados
qrs_detections = np.array(mat_struct["qrs_detections"])

# Definimos cantidad de realizaciones en función de la cantidad de latidos detectados
R = qrs_detections.size

# Definimos cantidad de muestras a tomar a izquierda y derecha para cada latido detectado
GAP1 = 250
GAP2 = 350

# Generamos una lista donde cada posición es un array de muestras de cada latido detectado
lista = [xx[int(qrs_detections[i]-GAP1):int(qrs_detections[i]+GAP2)] for i in range(0,R)]

# Generamos una matriz donde cada columna es un latido y cada fila es una muestra de cada latido
pulsos = np.hstack(lista)

# Quitamos el valor medio a las amplitudes de los latidos para alinearlos verticalmente
pulsos = pulsos - np.mean(pulsos,axis=0)

# Normalizamos amplitud de los latidos en función de la máxima amplitud en toda la matriz
pulsos = pulsos / np.max(pulsos)

# Graficamos los latidos detectados en el registro
plt.figure()
plt.plot(pulsos)
plt.title("Latidos presentes en el registro")
plt.ylabel("Amplitud normalizada")
plt.xlabel("Tiempo (ms)")
plt.ylim([-0.4,1])
plt.show()

# %% Clasificación de latidos

# Patrón de latidos normales y ventriculares
heartbeat_pattern1 = np.array(mat_struct["heartbeat_pattern1"])
heartbeat_pattern2 = np.array(mat_struct["heartbeat_pattern2"])

# Alineamos todas las amplitudes de forma tal que la primer muestra tenga aplitud 0
heartbeat_pattern1 = heartbeat_pattern1 - heartbeat_pattern1[0]
heartbeat_pattern2 = heartbeat_pattern2 - heartbeat_pattern2[0]

# Normalizamos la amplitud de los latidos patrones respecto a la máxima amplitud encontrada
heartbeat_pattern1 = heartbeat_pattern1 / np.max(heartbeat_pattern1)
heartbeat_pattern2 = heartbeat_pattern2 / np.max(heartbeat_pattern2)

# Aplanamos dimensión del patrón para poder aplicar padding
heartbeat_pattern1 = heartbeat_pattern1.flatten()
heartbeat_pattern2 = heartbeat_pattern2.flatten()

# Aplicamos padding a izquierda a los patrones, relllenando con el valor de la primer muestra
# Esto es requerido para que la dimensiones de los patrones coincidan con las dimensiones de los latidos y poder operar
heartbeat_pattern1 = np.pad(heartbeat_pattern1, (189, 0), 'constant', constant_values=(heartbeat_pattern1[0], 0))
heartbeat_pattern2 = np.pad(heartbeat_pattern2, (159, 0), 'constant', constant_values=(heartbeat_pattern2[0], 0))

# Definimos listas donde almacenamos los coeficientes calculados para cada modelo
corrs_norm = []
corrs_vent = []

# Cálculo de coeficientes de correlación para cada modelo y cada latido
for i in range(0,pulsos.shape[1]):
    pulso = pulsos[:,i].flatten()
    corr_norm = np.corrcoef(pulso,heartbeat_pattern1)
    corr_vent = np.corrcoef(pulso,heartbeat_pattern2)
    corrs_norm.append(corr_norm[0][1])
    corrs_vent.append(corr_vent[0][1])

# Transformamos las listas en numpy arrays
corrs_norm = np.array(corrs_norm)
corrs_vent = np.array(corrs_vent)
  
# Generamos indexador booleano para los latidos en base a los valores obtenidos de R
bcorrs1 = (corrs_norm > 0.1) & (corrs_norm > corrs_vent)
bcorrs2 = (corrs_vent > 0.8) & (corrs_vent > corrs_norm)

# Obtenemos los latidos normales y ventriculares en arrays separados a partir de indexador booleano
pulsos_normales      = pulsos[:,bcorrs1]
pulsos_ventriculares = pulsos[:,bcorrs2]

# %% Estimación espectral

from scipy import signal as sig

# Recuperamos los primeros 50 pulsos normales y ventriculares que tienen poco ruido, usando slicing
# NOTA: Es posible usar slicing dado que los indices de latidos detectados son crecientes (ordenados temporalmente)
pulsos_normales_puros      = pulsos_normales[:,:50]
pulsos_ventriculares_puros = pulsos_ventriculares[:,:50]

# Definimos la frecuencia de muestreo de nuestra señal
fs = 1000

# Para mejorar la visualización de nuestra DSP, aplicamos zero-padding a nuestra señal

# Definimos la cantidad de muestras a tomar para nuestra señal con padding (en ms)
N = 4000

# Calculamos la cantidad de padding a izquierda y derecha a aplicar a nuestra señal (es igual para normal y ventricular)
PAD = int((N - len(pulsos_normales_puros))/2)

# Aplicamos zero-padding a nuestras señales
pulsos_normales_puros_pad      = np.pad(pulsos_normales,      pad_width=((PAD,PAD), (0,0)), mode='constant')
pulsos_ventriculares_puros_pad = np.pad(pulsos_ventriculares, pad_width=((PAD,PAD), (0,0)), mode='constant')

# Calculamos el PSD de nuestras señales
f, psd_normales      = sig.welch(pulsos_normales_puros_pad,      fs=fs, nperseg=N/2,axis=0)
f, psd_ventriculares = sig.welch(pulsos_ventriculares_puros_pad, fs=fs, nperseg=N/2,axis=0)

# Normalizamos amplitud de las PSD
psd_normales      = psd_normales/np.max(psd_normales)
psd_ventriculares = psd_ventriculares/np.max(psd_ventriculares)

# Calculamos el PSD medio
psd_normales_avg      = np.mean(psd_normales,axis=1)
psd_ventriculares_avg = np.mean(psd_ventriculares,axis=1)

plt.close("all")

# Graficamos PSD de latidos normales
plt.figure()
plt.title("PSD (método de Welch) - Latidos normales")
plt.plot(f,psd_normales)
plt.plot(f,psd_normales_avg,"-kx",lw=1,label="PSD medio")
plt.ylabel("Amplitud normalizada")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0,35)
plt.legend()
plt.show()

# Graficamos PSD de latidos ventriculares
plt.figure()
plt.title("PSD (método de Welch) - Latidos ventriculares")
plt.plot(f,psd_ventriculares)
plt.plot(f,psd_ventriculares_avg,"-kx",lw=1,label="PSD medio")
plt.ylabel("Amplitud normalizada")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0,50)
plt.legend()
plt.show()

# %% Determinación de la plantilla de filtro

psd_norm_cumsum = np.cumsum(psd_normales_avg)/np.sum(psd_normales_avg)
psd_vent_cumsum = np.cumsum(psd_ventriculares_avg)/np.sum(psd_ventriculares_avg)

psd_umbral = 0.99

idx_norm_umbral = np.argmax(psd_norm_cumsum > psd_umbral)
idx_vent_umbral = np.argmax(psd_vent_cumsum > psd_umbral)

plt.close("all")

# Graficamos PSD de latidos normales
plt.figure()
plt.title("PSD (método de Welch) - Latidos normales")
plt.plot(f,psd_normales)
plt.plot(f,psd_normales_avg,"-kx",lw=1,label="PSD medio")
plt.axvline(f[idx_norm_umbral],ls="--",color="r")
plt.ylabel("Amplitud normalizada")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0,50)
plt.legend()
plt.show()

# Graficamos PSD de latidos ventriculares
plt.figure()
plt.title("PSD (método de Welch) - Latidos ventriculares")
plt.plot(f,psd_ventriculares)
plt.plot(f,psd_ventriculares_avg,"-kx",lw=1,label="PSD medio")
plt.axvline(f[idx_vent_umbral],ls="--",color="r")
plt.ylabel("Amplitud normalizada")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0,50)
plt.legend()
plt.show()

# %% Verificación de la frecuencia de corte

pulsos_norm_avg = np.mean(pulsos_normales_puros_pad,axis=1)
pulsos_vent_avg = np.mean(pulsos_ventriculares_puros_pad,axis=1)

N = len(pulsos_norm_avg)

fft_morm_avg = np.fft.fft(pulsos_norm_avg)
fft_vent_avg = np.fft.fft(pulsos_vent_avg)

df = fs / N

f2 = np.linspace(0,(N-1)*df,N)

bidx_norm = f2 >= f[idx_norm_umbral]
bidx_vent = f2 >= f[idx_vent_umbral]

fft_morm_avg[bidx_norm] = 0
fft_vent_avg[bidx_vent] = 0

fft_morm_abs = np.abs(fft_morm_avg)
fft_vent_abs = np.abs(fft_vent_avg)

fft_morm_abs = fft_morm_abs / np.max(fft_morm_abs)
fft_vent_abs = fft_vent_abs / np.max(fft_vent_abs)

plt.close("all")

# Graficamos FFT de latidos normales
plt.figure()
plt.title("FFT - Latidos normales")
plt.plot(f2,fft_morm_abs)
plt.ylabel("Amplitud normalizada")
plt.xlabel("Frecuencia (Hz)")
plt.show()

# Graficamos FFT de latidos ventriculares
plt.figure()
plt.title("FFT - Latidos ventriculares")
plt.plot(f2,fft_vent_abs)
plt.ylabel("Amplitud normalizada")
plt.xlabel("Frecuencia (Hz)")
plt.show()


norm_rec = np.fft.ifft(fft_morm_avg)
vent_rec = np.fft.ifft(fft_vent_avg)

# Graficamos FFT de latidos ventriculares
plt.figure()
plt.title("Latidos normales reconstruidos")
plt.plot(norm_rec)
plt.ylabel("Amplitud normalizada")
plt.xlabel("Tiempo (ms)")
plt.show()

# Graficamos FFT de latidos ventriculares
plt.figure()
plt.title("Latidos ventriculares reconstruidos")
plt.plot(vent_rec)
plt.ylabel("Amplitud normalizada")
plt.xlabel("Tiempo (ms)")
plt.show()
