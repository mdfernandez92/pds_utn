import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# SEÑALES SIN PAD

# fs = 1000
# N  = 1000

# ts = 1/fs

# df = fs/N

# tt = np.linspace(0, (N-1)*ts, N)

# ff = np.linspace(0, (N-1)*df, N)

# f0 = np.array([N/8, N/4, N*3/8])

# for ii in f0:
#     xx = np.hstack((
#         np.sin(2*np.pi*(ii)*df*tt).reshape(N,1),
#         np.sin(2*np.pi*(ii+0.25)*df*tt).reshape(N,1),
#         np.sin(2*np.pi*(ii+0.5)*df*tt).reshape(N,1)
#     ))
    
#     # Normalizamos potencia
#     xx = xx / np.sqrt(np.mean(xx**2, axis=0))
    
#     XX = 1/N * np.fft.fft(xx, axis = 0)
    
#     bfrec = ff <= fs/2
    
#     plt.figure()
#     plt.plot(ff[bfrec],10* np.log10(2*np.abs(XX[bfrec,:])**2),':x')
#     plt.ylim((-100,5))
#     plt.show()
    


fs = 1000
N  = 1000

ts = 1/fs

df = fs/N

tt = np.linspace(0, (N-1)*ts, N)

ff = np.linspace(0, (N-1)*df, N)

f0 = np.array([N/8, N/4, N*3/8])

cant_pad = 9

for ii in f0:
    xx = np.hstack((
        np.sin(2*np.pi*(ii)*df*tt).reshape(N,1),
        np.sin(2*np.pi*(ii+0.25)*df*tt).reshape(N,1),
        np.sin(2*np.pi*(ii+0.5)*df*tt).reshape(N,1)
    ))

    # Normalizamos potencia
    xx = xx / np.sqrt(np.mean(xx**2, axis=0))
            
    zz = np.zeros_like(xx)

    xx_pad = np.vstack([xx, zz.repeat(cant_pad, axis = 0)])

    N_Pad = xx_pad.shape[0]
        
    # df = fs/N_PAD
    # ff = np.linspace(0, (N_PAD-1)*df, N_PAD)
    
    
    XX = 1/N * np.fft.fft(xx_pad, axis = 0)
    
    # bfrec = ff <= fs/2
    
    # plt.figure()
    # plt.plot(ff[bfrec],10* np.log10(2*np.abs(XX[bfrec,:])**2),':x')
    # plt.ylim((-100,5))
    # plt.show()