# -*- coding: utf-8 -*-

# %% Generación de 


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from pandas import DataFrame
# from IPython.display import HTML

plt.close("all")

fs = 1000 # Hz

# Insertar aquí el código para inicializar tu notebook
########################################################

# Realizaciones
R = 200

# Simular para los siguientes tamaños de señal
N = np.array([10, 50, 100, 250, 500, 1000, 5000], dtype=int)

# Parametros para la señal aleatoria
mean = 0
σ2   = 2
std  = np.sqrt(σ2)

sesgo = np.zeros(N.size)
varianza = np.zeros(N.size)

##########################################
# Acá podés generar los gráficos pedidos #
##########################################
for i in range(0,len(N)):
    # Generamos la señal de ruido
    xx = np.random.normal(mean,std,(N[i],R))
    
    # Calculamos periodograma
    PP = 1/N[i] * np.abs(np.fft.fft(xx,axis=0))**2
    
    # Calculamos periodograma promedio
    PP_mean = np.mean(PP, axis=1)
    
    # Calculamos el sesgo
    sesgo[i] = np.mean(PP_mean) - 2
    
    # Calculamos la varianza del periodograma
    PP_var = np.var(PP, axis=1)
    
    # Calculamos la varianza promedio
    varianza[i] = np.mean(PP_var)
    
    
