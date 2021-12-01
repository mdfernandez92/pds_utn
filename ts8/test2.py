# -*- coding: utf-8 -*-

# Importamos bibliotecas a utilizar
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.close("all")

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
Ω1 *= norm

# Transformamos las dimensiones del array para poder multiplicar por tt
Ω1 = Ω1.reshape(1,R)

# Definimos la potencia de la señal senoidal como unitaria
pot_sen = 1

# Calculamos la amplitud de la señal senoidal a partir de la potencia (en veces)
a1 = np.sqrt(2*pot_sen)

# Generamos la señal senoidal
xx_sen = a1 * np.sin(2*np.pi * Ω1 * tt)

SNR = np.array([3, 10], dtype=float) #En dB

# %% PSD Senoidal

plt.close("all")

# Definimos padding a aplicar a mi señal senoidal
PAD = 0 * N

# Aplicamos zero-padding a nuestras señales
xx_sen_pad  = np.pad(xx_sen, pad_width=((0,PAD), (0,0)), mode='constant')

# Graficamos PSD
plt.figure()
plt.title("Señal senoidal con padding")
plt.plot(xx_sen_pad)
plt.ylabel("Amplitud normalizada")
plt.xlabel("Tiempo (s)")
# plt.xlim(0,45)
plt.show()

# Calculamos el PSD de la señal senoidal con padding (método de Welch)
f, psd_sen = sig.welch(xx_sen_pad, fs=fs, nperseg=(N+PAD)/2,axis=0)

# Normalizamos amplitud de la PSD
psd_sen = psd_sen / np.max(psd_sen,axis=0)

# Graficamos PSD
plt.figure()
plt.title("PSD (método de Welch) - Señal senoidal")
plt.plot(f,psd_sen,":x",lw=1)
plt.ylabel("Amplitud normalizada")
plt.xlabel("Frecuencia (Hz)")
# plt.xlim(0,45)
plt.show()

# Graficamos PSD
plt.figure()
plt.title("PSD (método de Welch) - Señal senoidal")
plt.plot(f,10*np.log10(psd_sen),"-",lw=1)
plt.ylabel("Amplitud normalizada [dB]")
plt.xlabel("Frecuencia (Hz)")
# plt.xlim(0,45)
plt.show()

# %% EXPERIMENTO 1 - POTENCIA RUIDO -3dB

pot_ruido = pot_sen/(10**(SNR[0]/10))

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
xx_sen_ruid = xx_sen + noise

# %% PSD Senoidal

plt.close("all")

# Aplicamos zero-padding a nuestras señales
xx_sen_ruid_pad  = np.pad(xx_sen_ruid, pad_width=((0,PAD), (0,0)), mode='constant')

# Aplicamos zero-padding a nuestras señales
xx_ruid_pad  = np.pad(noise, pad_width=((0,PAD), (0,0)), mode='constant')

# Calculamos el PSD de la señal senoidal con padding (método de Welch)
f, psd_sen_ruid = sig.welch(xx_sen_ruid_pad, fs=fs, nperseg=(N+PAD)/2,axis=0)

# Calculamos el PSD de la señal senoidal con padding (método de Welch)
f, psd_ruid = sig.welch(xx_ruid_pad, fs=fs, nperseg=(N+PAD)/2,axis=0)

# Normalizamos amplitud de la PSD
psd_sen_ruid = psd_sen_ruid / np.max(psd_sen_ruid)

# Para evitar aplicar logaritmo a amplitudes iguales a 0, vamos a sumar resolución de flotante a cada amplitud
eps = np.finfo(float).eps

# Normalizamos amplitud de la PSD
psd_ruid = psd_ruid / np.max(psd_sen_ruid,axis=0)

# # Graficamos PSD
# plt.figure()
# plt.title("PSD (método de Welch) - Señal senoidal")
# plt.plot(f,psd_sen_ruid,":x",lw=1)
# plt.ylabel("Amplitud normalizada")
# plt.xlabel("Frecuencia (Hz)")
# # plt.xlim(0,45)
# plt.show()

# Graficamos PSD
plt.figure()
plt.title("PSD (método de Welch) - Señal senoidal")
plt.plot(f,10*np.log10(psd_sen+eps),"-",lw=1)
plt.plot(f,10*np.log10(psd_ruid+eps),"-",lw=1)
plt.ylabel("Amplitud normalizada [dB]")
plt.xlabel("Frecuencia (Hz)")
# plt.xlim(0,45)
plt.show()
