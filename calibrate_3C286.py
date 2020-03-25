import numpy as np
import matplotlib.pylab as plt

nfreq = 1536
freq = (1219.70092773,1519.50561523)
freq = np.linspace(freq[0], freq[-1], nfreq)
alpha = 0.51 # spectral index of 3c286

indmn, indmx = 105000, -1

fn = '/data1/data/FRBs/FRB191108/iquv/CB21/polcal/stokes'

#/tank/data/FRBs/FRB200216/iquv/3C286/CB05/on/npy/stokesI_tab05_5.npy
fn = '/tank/data/FRBs/FRB200216/iquv/3C286/CB05/on/npy/stokes'
tab = 0
Ion = np.load(fn+'I_tab%0.2d_5.npy'%tab)[:, indmn:indmx]
Qon = np.load(fn+'Q_tab%0.2d_5.npy'%tab)[:, indmn:indmx]
Uon = np.load(fn+'U_tab%0.2d_5.npy'%tab)[:, indmn:indmx]
Von = np.load(fn+'V_tab%0.2d_5.npy'%tab)[:, indmn:indmx]

fn = '/tank/data/FRBs/FRB200216/iquv/3C286/CB05/off/npy/stokes'
Ioff = np.load(fn+'I_tab%0.2d_5.npy'%tab)[:, indmn:indmx]
Qoff = np.load(fn+'Q_tab%0.2d_5.npy'%tab)[:, indmn:indmx]
Uoff = np.load(fn+'U_tab%0.2d_5.npy'%tab)[:, indmn:indmx]
Voff = np.load(fn+'V_tab%0.2d_5.npy'%tab)[:, indmn:indmx]

#Ion = np.load(fn+'I_tab%0.2d.npy'%tab)[:, indmn:indmx]
#Qon = np.load(fn+'Q_tab%0.2d.npy'%tab)[:, indmn:indmx]
#Uon = np.load(fn+'U_tab%0.2d.npy'%tab)[:, indmn:indmx]
#Von = np.load(fn+'V_tab%0.2d.npy'%tab)[:, indmn:indmx]

#fn = '/tank/data/FRBs/FRB200216/iquv/3C286/CB05/off/npy/stokes'
#Ioff = np.load(fn+'I_tab%0.2d_off.npy'%tab)[:, indmn:indmx]
#Qoff = np.load(fn+'Q_tab%0.2d_off.npy'%tab)[:, indmn:indmx]
#Uoff = np.load(fn+'U_tab%0.2d_off.npy'%tab)[:, indmn:indmx]
#Voff = np.load(fn+'V_tab%0.2d_off.npy'%tab)[:, indmn:indmx]
fn = '/tank/data/FRBs/FRB200216/iquv/3C286/CB05/on/npy/stokes'
rfi_mask = []
ntime = len(Ion[0])

Ion = Ion[:, :ntime//100*100].reshape(nfreq, -1, 100).mean(-1).mean(-1)
Qon = Qon[:, :ntime//100*100].reshape(nfreq, -1, 100).mean(-1).mean(-1)
Uon = Uon[:, :ntime//100*100].reshape(nfreq, -1, 100).mean(-1).mean(-1)
Von = Von[:, :ntime//100*100].reshape(nfreq, -1, 100).mean(-1).mean(-1)

Ioff = Ioff[:, :ntime//100*100].reshape(nfreq, -1, 100).mean(-1).mean(-1)
Qoff = Qoff[:, :ntime//100*100].reshape(nfreq, -1, 100).mean(-1).mean(-1)
Uoff = Uoff[:, :ntime//100*100].reshape(nfreq, -1, 100).mean(-1).mean(-1)
Voff = Voff[:, :ntime//100*100].reshape(nfreq, -1, 100).mean(-1).mean(-1)

I = Ion - Ioff
Q = (Qon - Qoff)/I
U = (Uon - Uoff)/I
V = (Von - Voff)/I

xy = U + 1j*V

bandpass = I*(freq/1370.0)**alpha
#plt.plot(I)
#plt.plot(Q)
#plt.plot(U)
#plt.plot(V)
#plt.plot(np.sqrt(Q**2 + U**2 + V**2)/I, color='k')
#np.save('/tank/data/FRBs/FRB200216/iquv/3C286/CB05/on/npy/xy_phase_3c286_frequency.npy', np.angle(xy)
np.save(fn+'xy_phase_3c286_frequency.npy', np.angle(xy))
np.save(fn+'bandpass_from_3c286_alpha-0.54_CB05.npy', bandpass)
plt.figure(figsize=(6,6))
plt.subplot(211)
plt.plot(freq, np.angle(xy*np.exp(-1j)))
plt.ylabel('XY phase', fontsize=15)
plt.subplot(212)
plt.plot(freq, bandpass)
plt.ylabel('Bandpass (arb)', fontsize=15)
plt.xlabel('Freq', fontsize=15)
plt.show()
