# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pds import mi_funcion_sen

import numpy as np
from numpy.fft import fft

vmax = 1
dc = 0
fs = 1000
f0 = fs/4 
ph = 0 
N = 1000
    
df = 1/(N*(1/fs))

tt, xx = mi_funcion_sen(vmax, dc, f0, ph, N, fs)
tt, xx_2 = mi_funcion_sen(vmax, dc, f0-df, ph, N, fs)
tt, xx_3 = mi_funcion_sen(vmax, dc, f0+df, ph, N, fs)
tt, xx_4 = mi_funcion_sen(vmax, dc, f0+(df/2), ph, N, fs)
# tt, xx_5 = mi_funcion_sen(vmax, dc, f0+(df/4), ph, N, fs)
# tt, xx_6 = mi_funcion_sen(vmax, dc, ((N/4) + 0.25)*df, ph, N, fs)
# tt, xx_7 = mi_funcion_sen(vmax, dc, ((N/4) + 0.75)*df, ph, N, fs)

# Generamos base de frecuencias para graficar DFT
ff = np.linspace(0, (N-1), num=N) * df

bfrec = ff <= fs/2

XX = fft(xx)
XX_2 = fft(xx_2)
XX_3 = fft(xx_3)
XX_4 = fft(xx_4)
# XX_5 = fft(xx_5)
# XX_6 = fft(xx_6)
# XX_7 = fft(xx_7)

plt.close('all')

# plt.figure(0)
# plt.plot(tt,xx)
# plt.show()

plt.figure(1)
plt.plot(ff[bfrec],20 * np.log10(np.abs(XX[bfrec]/N)))
plt.plot(ff[bfrec],20 * np.log10(np.abs(XX_2[bfrec]/N)))
plt.plot(ff[bfrec],20 * np.log10(np.abs(XX_3[bfrec]/N)))
plt.plot(ff[bfrec],20 * np.log10(np.abs(XX_4[bfrec]/N)),"x:g")
# plt.plot(ff[bfrec],20 * np.log10(np.abs(XX_5[bfrec]/N)),"x:r")
# plt.plot(ff[bfrec],20 * np.log10(np.abs(XX_6[bfrec]/N)),"x:b")
# plt.plot(ff[bfrec],20 * np.log10(np.abs(XX_7[bfrec]/N)),"x:o")
plt.show()

# plt.figure(1)
# plt.plot(ff[bfrec],np.abs(XX[bfrec]/N))
# plt.show()


# plt.figure(2)
# plt.plot(ff[bfrec],np.abs(XX_2[bfrec]/N))
# plt.show()

# plt.figure(3)
# plt.plot(ff[bfrec],np.abs(XX_3[bfrec]/N))
# plt.show()



# TS5 probar lo que dice emanuel