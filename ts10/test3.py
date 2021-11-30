# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 00:51:14 2021

@author: Mauro
"""

# Importamos bibliotecas a utilizar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio
import scipy.signal as sig

# Configuramos tamaño de gráficos
mpl.rcParams['figure.figsize'] = (14,7)

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

# Para evitar aplicar logaritmo a amplitudes iguales a 0, vamos a sumar resolución de flotante a cada amplitud
eps = np.finfo(float).eps

idx_interf_low_freq = ( 
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )

idx_no_interf = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )



# Definimos plantilla del filtro
fs0 = 0.1
fc0 = 1
fc1 = 35
fs1 = 45
alfa_min = 60  # dB
alfa_max = 0.5 # dB

#-------------------------------------
# Filtro Pasa-Banda Butter
#-------------------------------------

# Definimos bandas de paso y corte (frec. normalizada a Nyquist)
wp = np.array([fc0, fc1]) / fnyq
ws = np.array([fs0, fs1]) / fnyq

# Diseñamos el filtro
sos_iir_bp_butter = sig.iirdesign(wp, ws, alfa_max, alfa_min/2, analog=False, ftype='butter', output='sos')

#-------------------------------------
# Filtro Pasa-Banda Butter
#-------------------------------------

# Definimos bandas de paso y corte (frec. normalizada a Nyquist)
wp = np.array([fc0, fc1]) / fnyq
ws = np.array([fs0, fs1]) / fnyq

# Diseñamos el filtro
sos_iir_bp_ellip = sig.iirdesign(wp, ws, alfa_max, alfa_min/2, analog=False, ftype='ellip',  output='sos')

#-------------------------------------------
# Filtro Pasa-Bajo Butter
#-------------------------------------------

# Definimos bandas de paso y corte (frec. normalizada a Nyquist)
wp = fc0 / fnyq
ws = 0.001 / fnyq

# Diseñamos el filtro
ba_lp_butter = sig.iirdesign(wp, ws, alfa_max, alfa_min/2, analog=False, ftype='butter', output='ba')

#-------------------------------------------
# Filtro Pasa-Alto Ellip
#-------------------------------------------

# Definimos bandas de paso y corte (frec. normalizada a Nyquist)
wp = fc1 / fnyq
ws = fs1 / fnyq

# Diseñamos el filtro
ba_hp_ellip = sig.iirdesign(wp, ws, alfa_max, alfa_min/2, analog=False, ftype='ellip', output='ba')

#--------------------------------------------------------
# Filtro Pasa-Banda = Pasa-Bajo Butter + Pasa-Alto Ellip
#--------------------------------------------------------

# Combinamos los filtros Pasa-Bajo y Pasa-Alto para formar un Pasa-Banda
sos_iir_bp_butter_ellip = sig.tf2sos(np.polymul(ba_lp_butter[0],ba_hp_ellip[0]),np.polymul(ba_lp_butter[1],ba_hp_ellip[1]))


#-------------------------------------------
# Filtro Pasa-Bajo Remez
#-------------------------------------------

# Definimos la cantidad de coeficientes
numtaps = 1501

# Definimos bandas de paso y corte
edges = np.array([0, fs0, fc0, fnyq])

# Definimos las ganancias en dB
gains = np.array([-200,0])

# Convertimos las ganancias a veces
gains = 10**(gains/20)

# Diseñamos el filtro
num_lp_remez = sig.remez(numtaps, edges, gains,fs=fs)

#-------------------------------------------
# Filtro Pasa-Alto Remez
#-------------------------------------------

# Definimos la cantidad de coeficientes
numtaps = 500

# Definimos bandas de paso y corte
edges = np.array([0, fc1, fs1, fnyq])

# Definimos las ganancias en dB
gains = np.array([0,-200])

# Convertimos las ganancias a veces
gains = 10**(gains/20)

# Diseñamos el filtro
num_hp_remez = sig.remez(numtaps, edges, gains,fs=fs)

#--------------------------------------------------------
# Filtro Pasa-Banda = Pasa-Bajo Remez + Pasa-Alto Remez
#--------------------------------------------------------

# Combinamos los filtros Pasa-Bajo y Pasa-Alto para formar un Pasa-Banda
num_fir_bp_remez = np.polymul(num_lp_remez,num_hp_remez)

#-------------------------------------------
# Filtro Pasa-Bajo Least Square
#-------------------------------------------

# Definimos la cantidad de coeficientes
numtaps = 1501

# Definimos bandas de paso y corte
edges = np.array([0, fs0, fc0, fnyq])

# Definimos las ganancias en dB
gains = np.array([-200,-(alfa_min),0,0])

# Convertimos las ganancias a veces
gains = 10**(gains/20)

# Diseñamos el filtro
num_lp_ls = sig.firls(numtaps, edges, gains,fs=fs)

#-------------------------------------------
# Filtro Pasa-Alto Least Square
#-------------------------------------------

# Definimos la cantidad de coeficientes
numtaps = 501

# Definimos bandas de paso y corte
edges = np.array([0, fc1, fs1, fnyq])

# Definimos las ganancias en dB
gains = np.array([0, 0, -(alfa_min), -200])

# Convertimos las ganancias a veces
gains = 10**(gains/20)

# Diseñamos el filtro
num_hp_ls = sig.firls(numtaps, edges, gains,fs=fs)

#--------------------------------------------------------
# Filtro Pasa-Banda = Pasa-Bajo LS + Pasa-Alto LS
#--------------------------------------------------------

# Combinamos los filtros Pasa-Bajo y Pasa-Alto para formar un Pasa-Banda
num_fir_bp_ls = np.polymul(num_lp_ls,num_hp_ls)


#--------------------------------------------------------
# Graficación de la respuestas en frecuencia
#--------------------------------------------------------

# Obtenemos las respuestas en frecuencia de los filtros
w, h_iir_bp_butter       = sig.sosfreqz(sos_iir_bp_butter,       fs=fs,worN=np.logspace(-2,2,2048))
w, h_iir_bp_ellip        = sig.sosfreqz(sos_iir_bp_ellip,        fs=fs,worN=w)
w, h_iir_bp_butter_ellip = sig.sosfreqz(sos_iir_bp_butter_ellip, fs=fs,worN=w)

w, h_fir_bp_remez = sig.freqz(num_fir_bp_remez, fs=fs,worN=w)
w, h_fir_bp_ls    = sig.freqz(num_fir_bp_ls,    fs=fs,worN=w)

plt.figure()
plt.title("Espectros de amplitud - Filtros IIR y FIR")
plt.plot(w,20*np.log10(np.abs(h_iir_bp_butter_ellip)+eps), label="IIR - Butter+Ellip ({:d})".format(sos_iir_bp_butter_ellip.shape[0] * 2))
plt.plot(w,20*np.log10(np.abs(h_iir_bp_ellip)+eps), label="IIR - Ellip ({:d})".format(sos_iir_bp_ellip.shape[0] * 2))
plt.plot(w,20*np.log10(np.abs(h_iir_bp_butter)+eps),label="IIR - Butter ({:d})".format(sos_iir_bp_butter.shape[0] * 2))
plt.plot(w,20*np.log10(np.abs(h_fir_bp_ls)+eps),label="FIR - Least Square ({:d})".format(num_fir_bp_ls.shape[0]))
plt.plot(w,20*np.log10(np.abs(h_fir_bp_remez)+eps),label="FIR - Remez ({:d})".format(num_fir_bp_remez.shape[0]))
plt.ylabel("|H(f)| (dB)")
plt.xlabel("Frecuencia (Hz)")
plt.ylim(-130,10)
plt.grid()
plt.legend()
plt.show()