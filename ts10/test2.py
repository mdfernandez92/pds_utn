# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 00:05:48 2021

@author: Mauro
"""

# Importamos bibliotecas a utilizar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio
import scipy.signal as sig

# Configuramos tamaño de gráficos
mpl.rcParams['figure.figsize'] = (10,5)

# Configuramos tamaño de fuentes de gráficos
plt.rcParams.update({'font.size': 12})

# Leemos el dataset desde el archivo ECG.mat
mat_struct = sio.loadmat("ecg.mat")

# Serie temporal con el registro del ECG
xx = mat_struct["ecg_lead"]

# Definimos la frecuencia de sampling de la señal
fs = 1000

# Calculamos frecuencia de Nyquist 
fnyq = 0.5 * fs

idx_interf_low_freq = ( 
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )

idx_no_interf = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

# Para evitar aplicar logaritmo a amplitudes iguales a 0, vamos a sumar resolución de flotante a cada amplitud
eps = np.finfo(float).eps

fnyq = 0.5 * fs

# %% FILTRO FIR

plt.close("all")

# Definimos plantilla de filtro Pasa-Banda
fs0 = 0.1
fc0 = 1
fc1 = 35
fs1 = 45
alfa_min = 60  # dB
alfa_max = 0.5 # dB

numtaps = 2000+1 # Size of the FIR filter.
edges = [0, fs0, fc0, fc1, fs1, fnyq]
gains = np.array([200, 0])
gains = 10**(gains/20)

taps = sig.remez(numtaps, edges, [0, 1, 0],fs=fs)
w, h_fir_remez = sig.freqz(taps, [1], worN=np.logspace(-2,2,2048))
h_abs_fir_remez = np.abs(h_fir_remez)

plt.figure()
plt.plot(w,20*np.log10(h_abs_fir_remez+eps),label="butter")
plt.grid()
plt.legend()
# plt.ylim(-130,10)
plt.show()

# %% Filtros IIR

plt.close("all")

# Definimos plantilla de filtro Pasa-Banda
fs0 = 0.1
fc0 = 1
fc1 = 35
fs1 = 45
alfa_min = 60  # dB
alfa_max = 0.5 # dB

# Definimos las bandas de paso y de corte (frecuencia normalizada a Nyquist)
wp = np.array([fc0, fc1]) / fnyq
ws = np.array([fs0, fs1]) / fnyq

# Diseñamos filtro IIR Butter SOS
sos_iir_butter = sig.iirdesign(wp, ws, alfa_max, alfa_min/2, analog=False, ftype='butter', output='sos')
sos_iir_ellip  = sig.iirdesign(wp, ws, alfa_max, alfa_min/2, analog=False, ftype='ellip',  output='sos')

# Obtenemos la respuesta en frecuencia del filtro
w, h_iir_butter = sig.sosfreqz(sos_iir_butter,fs=fs,worN=np.logspace(-2,2,2048))
w, h_iir_ellip  = sig.sosfreqz(sos_iir_ellip, fs=fs,worN=w)

plt.figure()
plt.plot(w,np.abs(h_iir_butter),label="butter (" + str(sos_iir_butter.shape[0] * 2) + ")")
plt.plot(w,np.abs(h_iir_ellip),label="ellip (" + str(sos_iir_ellip.shape[0] * 2) + ")")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(w,20*np.log10(np.abs(h_iir_butter)+eps),label="butter")
plt.plot(w,20*np.log10(np.abs(h_iir_ellip)+eps), label="ellip")
plt.grid()
plt.legend()
plt.show()