import numpy as np

def mi_cuantizador(sr,B,VF):
    """Cuantiza una señal discreta recibida como parámetro utilizando B bits en el rango VF
    
    Args:
        sr (float[nn]): señal a cuantizar, una matriz (Nx1) de números reales. 
        B  (int):       cantidad de Bits a utilizar para la cuantización
        VF (float):     valor absoluto del máximo valor del rango de amplitudes de la señal a cuantizar

    Returns:
        sq (float[nn]): señal cuantizada
    """
    # Calculamos paso de cuantización
    q = VF / (2**(B-1))
        
    # Calculamos cantidad de valores discretos
    N = (2**B)
        
    # Generamos la base de cuantización hasta N+1
    qq = np.linspace(-VF, VF, N+1, endpoint=True)
    
    # Obtenemos el valor medio entre cada rango discreto
    mm = (qq[1:] + qq[:-1])/2.0

    # Obtenemos el indice asociado a la base de cuantización
    idx = np.digitize(sr, mm)
    
    # Obtenemos el valor mas proximo cuantizado al de la señal
    sq = qq[idx]
    
    return sq

# Importamos desde biblioteca PDS generadores de señales realizados previamente 
from pds import mi_funcion_sen

# Importamos biblioteca para gráficos
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.close('all')

# Configuración del tamaño de fuente
fig_font_size = 14
plt.rcParams.update({'font.size':fig_font_size})

# Configuración del tamaño de los gráficos
mpl.rcParams['figure.figsize'] = (8,5)

# CUANTIZACION 4 BITS

# # Definimos parámetros de la señal senoidal
# vmax = 1    # Amplitud [V]
# dc   = 0    # Valor Medio [V]
# ph   = 0    # Fase [rad]
# N    = 1000 # N° muestras ADC
# fs   = 1000 # Frecuencia de muestreo ADC [Hz] 
# f0   = fs/N # Frecuencia [Hz]

# # Generamos señal utilizando los parámetros definidos
# tt, sr = mi_funcion_sen(vmax, dc, f0, ph, N, fs)

# # Definimos parámetros de cuantización 1
# B  = 4 # Cantidad de bits del ADC
# VF = 2 # Valor absoluto del valor máximo del rango [V]

# # Generamos señal cuantizada
# sq = mi_cuantizador(sr,B,VF)

# # Obtenemos el error de cuantización
# e = sq - sr 

# # Graficación de señal senoidal y su cuantización en el dominio del tiempo
# plt.figure(1);
# plt.plot(tt, sr, label='$S_R$')
# plt.plot(tt, sq, label='$S_Q$')
# plt.title('Cuantización de señal senoidal a 4 bits - 1Hz - 1 ciclo')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud [V]')
# plt.legend()
# plt.show()

# # Graficación del error de cuantización
# plt.figure(2)
# plt.stem(tt,e)
# plt.title('Errror de cuantización a 4 bits')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Error [V]')
# plt.legend()
# plt.show()


# PRUEBA RUIDO GAUSSIANO

# Definimos parámetros de la señal senoidal
vmax = 1    # Amplitud [V]
dc   = 0    # Valor Medio [V]
ph   = 0    # Fase [rad]
N    = 1000 # N° muestras ADC
fs   = 1000 # Frecuencia de muestreo ADC [Hz] 
f0   = fs/N # Frecuencia [Hz]

# # Calculamos el paso frecuencial
df = 1/(N*(1/fs))

# # Generamos base de frecuencias para graficar DFT
ff = np.linspace(0, (N-1), num=N) * df

# Generamos señal utilizando los parámetros definidos
tt, sr = mi_funcion_sen(vmax, dc, f0, ph, N, fs)

# Generamos ruido gaussiano
noise = np.random.normal(0,0.1,tt.size)

# Agregamos ruido a la señal senoidal
# sr += noise

# Definimos parámetros de cuantización 1
B  = 4 # Cantidad de bits del ADC
VF = 2 # Valor absoluto del valor máximo del rango [V]

# Calculamos paso de cuantización
q = VF / (2**(B-1))

# Generamos señal cuantizada
sq = mi_cuantizador(sr,B,VF)

# Obtenemos el error de cuantización
e = sq - sr 

# Calculamos autocorrelación del error
ac = np.correlate(e, e, mode = 'full')
ac = ac[ac.size // 2 :]

# Calculamos DEP
dep = np.fft.fft(ac)

valor_medio = np.mean(np.abs(dep))

# Area DEP desde FAC
area_dep_fac = np.sum(np.abs(dep))

# Area DEP desde e
area_dep_e = np.sum(np.abs(np.fft.fft(e))**2)

# Valor medio DEP desde FAC
h_dep_fac = area_dep_fac / dep.size # densidad promedio

# Valor medio DEP desde e
h_dep_e =  area_dep_e / dep.size

# Calculamos fft de la señal ruidosa cuantizada
X_sq = np.fft.fft(sq)

# Graficación de señal senoidal y su cuantización en el dominio del tiempo
plt.figure(1);
plt.plot(tt, sr, label='$S_R$')
plt.plot(tt, sq, label='$S_Q$')
plt.title('Cuantización de señal senoidal a 4 bits - 1Hz - 1 ciclo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.ylim(-2,2)
plt.legend()
plt.show()

# Graficación del error de cuantización
plt.figure(2)
plt.stem(tt,e)
plt.title('Errror de cuantización a 4 bits')
plt.xlabel('Tiempo [s]')
plt.ylabel('Error [V]')
plt.ylim(-q,q)
plt.show()

# Graficación del error de cuantización
plt.figure(3)
plt.hist(e,bins="auto")
plt.title('Histograma a 4 bits')
plt.show()

# Graficación del error de cuantización
plt.figure(4)
plt.stem(ac)
plt.title('Autocorrelación a 4 bits')
plt.xlabel('')
plt.ylabel('Amplitud [V]')
plt.show()

# Graficación del error de cuantización
plt.figure(5)
plt.stem(np.abs(dep))
plt.title('DEP a 4 bits')
plt.xlabel('')
plt.ylabel('Amplitud [V]')
plt.show()

# Graficación del error de cuantización
plt.figure(6)
plt.stem(np.abs(np.fft.fft(e))**2)
plt.title('DEP a 4 bits')
plt.xlabel('')
plt.ylabel('Amplitud [V]')
plt.show()

# Graficación del error de cuantización
plt.figure(7)
plt.stem(ff,np.abs(X_sq))
plt.title('FFT de la señal ruidosa cuantizada a 4 bits')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X_sq(f)|')
plt.xlim(0,20)
plt.show()

# Graficación del error de cuantización [DB]
plt.figure(8)
plt.stem(ff,10*np.log10((np.abs(X_sq)/N)**2))
plt.title('FFT de la señal ruidosa cuantizada a 4 bits')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X_sq(f)| [dbW]')
plt.xlim(0,100)
plt.show()



# # Definimos parámetros de cuantización 2
# B  = 8 # Cantidad de bits del ADC
# VF = 2 # Valor absoluto del valor máximo del rango [V]

# # Generamos señal cuantizada
# sq = mi_cuantizador(sr,B,VF)

# # Graficación de señal en el dominio del tiempo
# plt.figure(2);
# plt.plot(tt, sr, label='$S_R$')
# plt.plot(tt, sq, label='$S_Q$')
# plt.title('Cuantización de señal senoidal a 8 bits - 1Hz - 1 ciclo')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud [V]')
# plt.ylim(-2,2)
# plt.legend()
# plt.show()

# # Definimos parámetros de cuantización 3
# B  = 16 # Cantidad de bits del ADC
# VF = 2  # Valor absoluto del valor máximo del rango [V]

# # Generamos señal cuantizada
# sq = mi_cuantizador(sr,B,VF)



    
# # Graficación de señal en el dominio del tiempo
# plt.figure(3);
# plt.plot(tt, sr, label='$S_R$')
# plt.plot(tt, sq, label='$S_Q$')
# plt.title('Cuantización de señal senoidal a 16 bits  - 1Hz - 1 ciclo')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud [V]')
# plt.ylim(-2,2)
# plt.legend()
# plt.show()