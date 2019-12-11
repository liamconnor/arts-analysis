import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pylab as plt

try:
	import simpulse
except:
	print("Could not import simpulse")

fn = '/tank/data/FRBs/FRB191108/iquv/CB21/numpyarr/SB37_dedispersed.npy'
arr = np.load(fn)

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_width_freq(data, nfreq=16, dt=8.192e-5, freq=(1219.70092773, 1519.50561523), plot=False):
	nf,nt = data.shape
	if nf>nfreq:
		data = data[:nf//nfreq*nfreq].reshape(nfreq, nf//nfreq, nt).mean(1)

	t = np.linspace(0,nt*dt*1e3,nt)
	tt = np.linspace(0,nt*dt*1e3,100*nt)
	freq_arr = np.linspace(freq[0], freq[-1], nfreq)
	sig_arr=[]
	if plot:
		fig = plt.figure(figsize=(11,11))

	for ii in range(nfreq):
		D = data[ii]
		D -= np.median(D)
		D /= np.max(D)
		popt,pcov = curve_fit(gaus,t,D,p0=[1,t[-1]/2.0,0.5]) 
		sig_arr.append(np.abs(2.35*popt[2])) 
		if plot:
			plt.subplot(4,4,ii+1)
			plt.plot(t, D, color='k')
			plt.plot(tt, gaus(tt, popt[0], popt[1], popt[2]), color='red')
			plt.legend([str(int(freq_arr[ii]))+' MHz'], fontsize=8)     
			plt.axis('off')
			plt.title("FWHM: %0.2fms" % (2.35*popt[2])) 

	return freq_arr, sig_arr

def fit_sim_frb(dm=1000.0, nfreq=1536, ntime=10000):
	data_event = np.zeros([nfreq, ntime])
	noise_event = np.random.normal(128, noise_std, nfreq*ntime).reshape(nfreq, ntime)
	sp = simpulse.single_pulse(NTIME, NFREQ, freq_arr.min(), freq_arr.max(),
                           dm, scat_tau_ref, width_sec, fluence,
                           spec_ind, 0.)
	sp.add_to_timestream(data_event, 0.0, NTIME*dt)
	data_event = data_event[::-1]
	print(data_event.shape)

	f,s = fit_width_freq(data_event)
	plt.plot(f,s)
	plt.show()