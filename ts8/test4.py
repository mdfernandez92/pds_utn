
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.close("all")

fs = 1000
N  = 1000

ts = 1/fs

tt = np.linspace(0,(N-1)*ts,N)

w0 = np.pi/2
fr = 0

w1 = w0 + fr * (2*np.pi)/N

# Definimos parámetro normalizador
norm = (fs / (2*np.pi))

# Desnormalizamos
f0 = w1 * norm

pot_sen = 1

A = np.sqrt(2*pot_sen)

xx = A * np.sin(2*np.pi * f0 * tt)

plt.figure()
plt.plot(tt,xx)
plt.show()

# %% Espectro de amplitud

xx_fft = 1/N * np.fft.fft(xx)

df = fs/N

ff = np.linspace(0,(N-1)*df,N)

bfrec = ff <= fs/2

plt.figure()
plt.plot(ff[bfrec],np.abs(xx_fft[bfrec])**2,"--x")
plt.show()

# %% Welch

PAD = N * 19

xx_pad = np.pad(xx,(0,PAD),mode="constant", constant_values=(0, 0))

N_pad = xx_pad.shape[0]

# Calculamos el PSD de nuestras señales
f, psd = sig.welch(xx_pad, fs=fs, nperseg=N_pad/2)

plt.figure()
plt.plot(f,psd,"--x")
plt.show()

# %% ruido

# Calculamos la potencia de la señal senoidal en dB
pot_sen_db = 10*np.log10(pot_sen)

# Calculamos la potencia del ruido en dB
pot_ruido_db = pot_sen_db - 30

# Calculamos la potencia del ruido en veces
pot_ruido = 10**(pot_ruido_db/10)

# Definimos valor medio y desvío estandard del ruido
# El desvío standard es la raiz de la varianza del ruido y la varianza es la potencia del mismo
noise_mean = 0
noise_var  = pot_ruido
noise_std  = np.sqrt(noise_var)

# Generamos la señal de ruido
noise = np.random.normal(noise_mean,noise_std,N)

plt.figure()
plt.plot(noise)
plt.show()

# %% señal ruidosa

# Generamos la señal final a partir de la senoidal y el ruido
xx_sen_ruid = xx + noise

plt.figure()
plt.plot(xx_sen_ruid)
plt.show()

# %% Welch

PAD = N * 19

xx_sen_ruid_pad = np.pad(xx_sen_ruid,(0,PAD),mode="constant", constant_values=(0, 0))

N_pad = xx_sen_ruid_pad.shape[0]

# Calculamos el PSD de nuestras señales
f, psd = sig.welch(xx_sen_ruid_pad, fs=fs, nperseg=N_pad/2)

plt.figure()
plt.plot(f,10*np.log10(psd/np.max(psd)),"--x")
plt.show()
