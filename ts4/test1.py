import matplotlib.pyplot as plt

# Importamos herramientas anteriormente implementadas de biblioteca PDS
from pds import mi_funcion_sen, mi_cuantizador, mi_cuantizador_2

vmax = 1
dc   = 0
fs   = 1000
N    = 1000
f0   = fs / N
ph   = 0

tt, x = mi_funcion_sen(vmax, dc, f0, ph, N, fs)

B  = 4
Vf = 2 

#sq_1 = mi_cuantizador(x,B,Vf)
sq_2 = mi_cuantizador_2(x,B,Vf)
    
plt.close('all')

plt.figure(1)
plt.plot(tt,x)
#plt.plot(tt,sq_1)
plt.plot(tt,sq_2)
plt.show()