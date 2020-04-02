import os

import numpy as np
import matplotlib.pylab as plt

ntime, nfreq = 1000, 1536
nstokes = 4

RM = 1000.0
xy_phase = np.random.uniform(0,2*np.pi)
stokes_name = ['I','Q','U','V']
stokes_val = [1.0, 0.25, 0.11, 0.0]
lam_arr = 3e2 / np.linspace(1219.7, 1519.5, nfreq)

data = np.zeros([nstokes,nfreq,ntime])
noise = np.random.normal(0,0.1,nstokes*ntime*nfreq).reshape(nstokes, nfreq,ntime)
data += noise

# Add polarization data to noise
for ii in range(nstokes):
    data[ii, :, ntime//2-5:ntime//2+5] += stokes_val[ii]

P = data[1] + 1j*data[2]
P *= (np.exp(2j*RM*lam_arr**2))[:, None]
data[1], data[2] = P.real, P.imag
XY = data[2] + 1j*data[3] 
XY *= np.exp(1j*xy_phase)
data[2], data[3] = XY.real, XY.imag

for ii in range(4):
    plt.subplot(2,2,ii+1)
    plt.imshow(data[ii], aspect='auto')
    plt.ylabel(stokes_name[ii])
    
plt.tight_layout()
plt.show()

np.save('./tests/data/poldata/numpyarr/testFRB_dedisp', data)
os.system('touch ./tests/data/poldata/numpyarr/DM0.0_SNR10_CB01_SB35_Width1.txt')
os.system('python polcal_script.py -d ./tests/data/poldata/ -g -pd -F')
