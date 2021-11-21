# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:03:10 2021

@author: Mauro
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio

plt.close("all")

# Configuramos tamaño de gráficos
mpl.rcParams['figure.figsize'] = (12,7)

# Configuramos tamaño de fuentes de gráficos
plt.rcParams.update({'font.size': 16})

mat_struct = sio.loadmat("ecg.mat")

xx = mat_struct["ecg_lead"]
qrs_detections = np.array(mat_struct["qrs_detections"])
heartbeat_pattern1 = np.array(mat_struct["heartbeat_pattern1"])
heartbeat_pattern2 = np.array(mat_struct["heartbeat_pattern2"])

plt.figure()
plt.plot(xx)
plt.ylabel("x(t)")
plt.xlabel("Tiempo [s]")
plt.show()

heartbeat_pattern1 = heartbeat_pattern1 - heartbeat_pattern1[0]
heartbeat_pattern2 = heartbeat_pattern2 - heartbeat_pattern2[0]

heartbeat_pattern1 = heartbeat_pattern1 / np.max(heartbeat_pattern1)
heartbeat_pattern2 = heartbeat_pattern2 / np.max(heartbeat_pattern2)

heartbeat_pattern1 = heartbeat_pattern1.flatten()
heartbeat_pattern2 = heartbeat_pattern2.flatten()

heartbeat_pattern1 = np.pad(heartbeat_pattern1, (189, 0), 'constant', constant_values=(heartbeat_pattern1[0], 0))
heartbeat_pattern2 = np.pad(heartbeat_pattern2, (159, 0), 'constant', constant_values=(heartbeat_pattern2[0], 0))

R = qrs_detections.size

GAP1 = 250
GAP2 = 350

lista = [xx[int(qrs_detections[i]-GAP1):int(qrs_detections[i]+GAP2)] for i in range(0,R)]

pulsos = np.hstack(lista)

pulsos = pulsos - np.mean(pulsos,axis=0)

pulsos = pulsos / np.max(pulsos)

plt.figure()
plt.plot(pulsos)
plt.title("Latidos presentes en el registro")
plt.ylabel("Amplitud normalizada")
plt.xlabel("Tiempo (ms)")
plt.ylim([-0.4,1])
plt.show()

plt.figure()
plt.title("Latido normal modelo (con padding)")
plt.plot(heartbeat_pattern1)
plt.ylabel("Amplitud normalizada")
plt.xlabel("Tiempo (ms)")
plt.show()

plt.figure()
plt.title("Latido ventricular modelo (con padding)")
plt.plot(heartbeat_pattern2)
plt.ylabel("Amplitud normalizada")
plt.xlabel("Tiempo (ms)")
plt.show()

# Correlacion 1

corrs1 = []

for i in range(0,pulsos.shape[1]):
    pulso = pulsos[:,i].flatten()    
    corr = np.corrcoef(pulso,heartbeat_pattern1)
    corrs1.append(corr[0][1])

corrs1 = np.array(corrs1)
    
# Correlacion 2

corrs2 = []

for i in range(0,pulsos.shape[1]):
    pulso = pulsos[:,i].flatten()    
    corr = np.corrcoef(pulso,heartbeat_pattern2)
    corrs2.append(corr[0][1])

corrs2 = np.array(corrs2)
    
bcorrs1 = (corrs1 > 0.1) & (corrs1 > corrs2)
bcorrs2 = (corrs2 > 0.8) & (corrs2 > corrs1)

pulsos_normales      = pulsos[:,bcorrs1]
pulsos_ventriculares = pulsos[:,bcorrs2]

plt.figure()
plt.title("Latidos presentes en el registro agrupados por tipo")
plt.plot(pulsos_normales,'g-', label='Normales')
plt.plot(pulsos_ventriculares,'b-', label='Ventriculares')
plt.ylabel("Amplitud normalizada")
plt.xlabel("Tiempo (ms)")
plt.ylim([-0.4,1])
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc='best')
plt.show()


plt.figure()
plt.title("Latidos promedio")
plt.plot(np.mean(pulsos_normales,axis=1),'g-', label='Normal')
plt.plot(np.mean(pulsos_ventriculares,axis=1),'b-', label='Ventricular')
plt.ylabel("Amplitud normalizada")
plt.xlabel("Tiempo (ms)")
plt.ylim([-0.4,1])
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc='best')
plt.show()





N = len(pulsos)

fs = 1000

df = fs/N

ff = np.linspace(0,(N-1)*df,N)

bfrec = ff <= fs/2

fftx_normales      = (1/N) * np.fft.fft(pulsos_normales,axis=0)
fftx_ventriculares = (1/N) * np.fft.fft(pulsos_ventriculares,axis=0)

fftx_abs_normales      = np.abs(fftx_normales)
fftx_abs_ventriculares = np.abs(fftx_ventriculares)

plt.figure()
plt.plot(ff[bfrec],fftx_abs_normales[bfrec]/np.max(fftx_abs_normales[bfrec],axis=0),'g-', label='Normales')
plt.plot(ff[bfrec],fftx_abs_ventriculares[bfrec]/np.max(fftx_abs_ventriculares[bfrec],axis=0),'b-', label='Ventriculares')
plt.title("Espectro de los latidos agrupados por tipo")
plt.ylabel("Amplitud normalizada")
plt.xlabel("Frecuencia [Hz]")
plt.xlim(0,100)
plt.ylim(-0.05,1.05)
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc='best')
plt.show()

plt.figure()
plt.scatter(corrs1,corrs2)
plt.xlabel("Modelo Normal")
plt.ylabel("Modelo Ventricular")
plt.show()

# pulsos_sel = pulsos[10,:]

# plt.figure()
# plt.plot(pulsos_sel)
# plt.show()

# psd_normales      = fftx_abs_normales[bfrec]**2
# psd_ventriculares = fftx_abs_ventriculares[bfrec]**2

# psd_norm_normales      = psd_normales      / np.max(psd_normales,axis=0)
# psd_norm_ventriculares = psd_ventriculares / np.max(psd_ventriculares,axis=0)

# # Para evitar aplicar logaritmo a amplitudes iguales a 0, vamos a sumar resolución de flotante a cada amplitud
# eps = np.finfo(float).eps

# psd_norm_normales      += eps
# psd_norm_ventriculares += eps

# plt.figure()
# plt.title("Densidad espectral de potencia de los latidos agrupados por tipo")
# plt.plot(ff[bfrec],10*np.log10(psd_norm_normales),     'g-', label='Normales')
# plt.plot(ff[bfrec],10*np.log10(psd_norm_ventriculares),'b-', label='Ventriculares')
# plt.ylabel("Amplitud normalizada [dB]")
# plt.xlabel("Frecuencia [Hz]")
# plt.xlim(0,100)
# plt.ylim(-80,0)
# handles, labels = plt.gca().get_legend_handles_labels()
# labels, ids = np.unique(labels, return_index=True)
# handles = [handles[i] for i in ids]
# plt.legend(handles, labels, loc='best')
# plt.show()

# from scipy import signal

# pulsos_normales_pad = np.pad(pulsos_normales, ((2000,2000),(0,0)), 'constant')
# pulsos_ventriculares_pad = np.pad(pulsos_ventriculares, ((2000,2000),(0,0)), 'constant')

# N = pulsos_normales_pad.shape[0]

# fs = 1/N

# f, psd_normales      = signal.welch(pulsos_normales_pad,      fs=fs, nperseg=N/2, axis=0)
# f, psd_ventriculares = signal.welch(pulsos_ventriculares_pad, fs=fs, nperseg=N/2, axis=0)


# plt.figure()
# plt.title("Densidad espectral de potencia de los latidos agrupados por tipo")
# plt.plot(f,psd_normales,     'g-', label='Normales')
# plt.plot(f,psd_ventriculares,'b-', label='Ventriculares')
# plt.ylabel("Amplitud")
# plt.xlabel("Frecuencia [Hz]")
# # plt.xlim(0,100)
# # plt.ylim(-80,0)
# handles, labels = plt.gca().get_legend_handles_labels()
# labels, ids = np.unique(labels, return_index=True)
# handles = [handles[i] for i in ids]
# plt.legend(handles, labels, loc='best')
# plt.show()


# nperseg = N/10

# f_normales        = []
# f_ventriculares   = []
# psd_normales      = []
# psd_ventriculares = []

# for i in range(0,pulsos_normales.shape[1]):
#     pulso_normal = pulsos_normales[:,i].flatten()
#     f_normal, psd_normal = signal.welch(pulso_normal, fs, nperseg=nperseg)
#     f_normales.append(f_normal)
#     psd_normales.append(psd_normal)

# for i in range(0,pulsos_ventriculares.shape[1]):
#     pulso_ventricular = pulsos_ventriculares[:,i].flatten()
#     f_ventricular, psd_ventricular = signal.welch(pulso_ventricular, fs, nperseg=nperseg)
#     f_ventriculares.append(f_ventricular)
#     psd_ventriculares.append(psd_ventricular)

# f_normales = np.vstack(f_normales).transpose()
# f_ventriculares = np.vstack(f_ventriculares).transpose()

# psd_normales = np.vstack(psd_normales).transpose()
# psd_ventriculares = np.vstack(psd_ventriculares).transpose()

# plt.figure()
# plt.title("Densidad espectral de potencia de los latidos agrupados por tipo")
# plt.plot(psd_normales,     'g-', label='Normales')
# plt.plot(psd_ventriculares,'b-', label='Ventriculares')
# plt.ylabel("Amplitud normalizada [dB]")
# plt.xlabel("Frecuencia [Hz]")
# # plt.xlim(0,100)
# # plt.ylim(-80,0)
# handles, labels = plt.gca().get_legend_handles_labels()
# labels, ids = np.unique(labels, return_index=True)
# handles = [handles[i] for i in ids]
# plt.legend(handles, labels, loc='best')
# plt.show()



# psd_norm_normales      = psd_normales      / np.max(psd_normales,axis=0)
# psd_norm_ventriculares = psd_ventriculares / np.max(psd_ventriculares,axis=0)

# # Para evitar aplicar logaritmo a amplitudes iguales a 0, vamos a sumar resolución de flotante a cada amplitud
# eps = np.finfo(float).eps

# psd_norm_normales      += eps
# psd_norm_ventriculares += eps


# from spectrum import WelchPeriodogram, marple_data
# psd = WelchPeriodogram(marple_data, 256)
