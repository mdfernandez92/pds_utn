# %% Generaciónn de datos

# Importamos bibliotecas a utilizar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pds import window_rectangular, window_bartlett, window_hann, window_blackman, window_flat_top

plt.close("all")

# Configuramos tamaño de gráficos
# mpl.rcParams['figure.figsize'] = (17,5)

# Configuramos tamaño de fuentes de gráficos
# plt.rcParams.update({'font.size': 12})

# Definimos parámetros de la señal senoidal a generar
N  = 1000 # Cantidad de muestras
fs = 1000 # Frecuencia de muestreo
a0 = 2    # Amplitud de la señal

# Calculamos el tiempo de muestreo
ts = 1/fs

# Generamos la base de tiempos
tt = np.linspace(0, (N-1)*ts, num=N).reshape(N,1)

# Definimos cantidad de realizaciones
R = 200

# Obtenemos las realizaciones de la variable aleatoria
fr = np.random.uniform(-2,2,R)

# Calculamos la frecuencia angular de la señal
Ω0 = np.pi / 2
Ω1 = Ω0 + fr * ((2*np.pi)/N)

# Definimos parámetro normalizador
norm = (fs / (2*np.pi))

# Desnormalizamos la frecuencia angular
Ω1 *= norm

# Transformamos las dimensiones del array para poder multiplicar por tt
Ω1 = Ω1.reshape(1,R)

# Generamos la amplitud de la señal para cada instante en la base de tiempos
xx = a0 * np.sin(2*np.pi * Ω1 * tt)

# Generamos las ventanas a aplicar a la señal
ww_rectangular = window_rectangular(N)
ww_bartlett    = window_bartlett(N)
ww_hann        = window_hann(N)
ww_blackman    = window_blackman(N)
ww_flat_top    = window_flat_top(N)

# Transformamos las dimensiones del array para poder multiplicar por la señal
ww_rectangular = ww_rectangular.reshape(N,1)
ww_bartlett    = ww_bartlett.reshape(N,1)
ww_hann        = ww_hann.reshape(N,1)
ww_blackman    = ww_blackman.reshape(N,1)
ww_flat_top    = ww_flat_top.reshape(N,1)

# Aplicamos las ventanas a la señal
xx_rectangular = xx * ww_rectangular
xx_bartlett    = xx * ww_bartlett
xx_hann        = xx * ww_hann
xx_blackman    = xx * ww_blackman
xx_flat_top    = xx * ww_flat_top

# Calculamos resolución espectral
df = ((2*np.pi)/N) * norm

# Generamos la base de frecuencias
#ff = np.linspace(0, (N-1)*df, num=N)
ff = np.fft.fftfreq(N,d=1/fs)

# Calculamos la fft de cada señal a la cual aplique una ventana
fftx_rectangular = np.fft.fft(xx_rectangular, axis=0) * (1/N) 
fftx_bartlett    = np.fft.fft(xx_bartlett,    axis=0) * (1/N) 
fftx_hann        = np.fft.fft(xx_hann,        axis=0) * (1/N) 
fftx_blackman    = np.fft.fft(xx_blackman,    axis=0) * (1/N) 
fftx_flat_top    = np.fft.fft(xx_flat_top,    axis=0) * (1/N) 

# Definimos la frecuencia desnormalizada asociada a mi estimador
f0 = Ω0 * norm

# Calculamos el valor de mi estimador para cada fft
a0_hat_rectangular = np.abs(fftx_rectangular[ff == f0, :])
a0_hat_bartlett    = np.abs(fftx_bartlett   [ff == f0, :])
a0_hat_hann        = np.abs(fftx_hann       [ff == f0, :])
a0_hat_blackman    = np.abs(fftx_blackman   [ff == f0, :])
a0_hat_flat_top    = np.abs(fftx_flat_top   [ff == f0, :])

# Graficamos histogramas
plt.figure()
plt.hist(a0_hat_rectangular.flatten(),label="Rectangular",bins=20)
plt.hist(a0_hat_bartlett.flatten(),label="Bartlett",bins=20)
plt.hist(a0_hat_hann.flatten(),label="Hann",bins=20)
plt.hist(a0_hat_blackman.flatten(),label="Blackman",bins=20)
plt.hist(a0_hat_flat_top.flatten(),label="Flat-top",bins=20)
plt.title("Histograma de las ventanas aplicadas")
plt.ylabel("Frecuencia")
plt.xlabel("Amplitudes estimadas del espectro")
plt.legend()
plt.show()

# %%

# Calculamos valor esperado, aproximado como la media muestral
mean_a0_hat_rectangular = np.mean(a0_hat_rectangular)
mean_a0_hat_bartlett    = np.mean(a0_hat_bartlett   )
mean_a0_hat_hann        = np.mean(a0_hat_hann       )
mean_a0_hat_blackman    = np.mean(a0_hat_blackman   )
mean_a0_hat_flat_top    = np.mean(a0_hat_flat_top   )

# Calculamos el sesgo
sa_rectangular = mean_a0_hat_rectangular - a0
sa_bartlett    = mean_a0_hat_bartlett    - a0
sa_hann        = mean_a0_hat_hann        - a0
sa_blackman    = mean_a0_hat_blackman    - a0
sa_flat_top    = mean_a0_hat_flat_top    - a0

# Calculamos la varianza
va_rectangular = np.var(a0_hat_rectangular)
va_bartlett    = np.var(a0_hat_bartlett   )
va_hann        = np.var(a0_hat_hann       )
va_blackman    = np.var(a0_hat_blackman   )
va_flat_top    = np.var(a0_hat_flat_top   )

# Presentamos los resultados en una tabla
data = [
    [sa_rectangular, va_rectangular],
    [sa_bartlett   , va_bartlett   ],
    [sa_hann       , va_hann       ],
    [sa_blackman   , va_blackman   ],
    [sa_flat_top   , va_flat_top   ]
]

df = pd.DataFrame(data, columns=['$s_a$', '$v_a$'], index=[
    'Rectangular',
    'Bartlett',
    'Hann',
    'Blackman',
    'Flat-top'
])

HTML(df.to_html())