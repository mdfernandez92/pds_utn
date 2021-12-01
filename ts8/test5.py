# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 00:44:43 2021

@author: Mauro
"""

# %% Generación de la señal
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
# f0 = w1 * norm
f0 = fs/4

pot_sen = 1

A = np.sqrt(2*pot_sen)

xx = A * np.sin(2*np.pi * f0 * tt)

# plt.figure()
# plt.plot(tt,xx)
# plt.show()

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
noise = np.random.normal(noise_mean,np.sqrt(noise_var),N)

plt.figure()
plt.plot(noise)
plt.show()

# %% PSD

xx_fft = 1/N * np.fft.fft(xx)
rr_fft = 1/N * np.fft.fft(noise)

df = fs/N

ff = np.linspace(0,(N-1)*df,N)

bfrec = ff <= fs/2

xx_fft_abs = 2*np.abs(xx_fft[bfrec])**2
rr_fft_abs = 2*np.abs(rr_fft[bfrec])**2


xx_fft_abs = xx_fft_abs / np.max(xx_fft_abs)
rr_fft_abs = rr_fft_abs / np.max(xx_fft_abs)

# Calculamos el valor medio del ruido
rr_mean = np.mean(rr_fft_abs)

plt.figure()
plt.plot(ff[bfrec],10*np.log10(xx_fft_abs))
plt.plot(ff[bfrec],10*np.log10(rr_fft_abs))
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(np.array([rr_mean, rr_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* rr_mean)) )
    
# plt.plot(ff[bfrec],10*np.log10(rr_mean),'--c')
plt.show()

