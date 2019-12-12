import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pylab as plt

import tools 

try:
	import simpulse
except:
	print("Could not import simpulse")

fn = '/tank/data/FRBs/FRB191108/iquv/CB21/numpyarr/SB37_dedispersed.npy'
arr = np.load(fn)

def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_width_freq(data, nfreq=16, dt=8.192e-5, freq=(1219.70092773, 1519.50561523), mk_plot=False):
	nf,nt = data.shape
	if nf>nfreq:
		data = data[:nf//nfreq*nfreq].reshape(nfreq, nf//nfreq, nt).mean(1)

	t = np.linspace(0,nt*dt*1e3,nt)
	tt = np.linspace(0,nt*dt*1e3,100*nt)
	freq_arr = np.linspace(freq[0], freq[-1], nfreq)
	sig_arr=[]
	if mk_plot:
		fig = plt.figure(figsize=(11,11))

	for ii in range(nfreq):
		D = data[ii]
		D -= np.median(D)
		D /= np.max(D)
		popt,pcov = curve_fit(gauss,t,D,p0=[1,t[-1]/2.0,0.5]) 
		sig_arr.append(np.abs(2.35*popt[2])) 
		if mk_plot:
			plt.subplot(4,4,ii+1)
			plt.plot(t, D, color='k')
			plt.plot(tt, gauss(tt, popt[0], popt[1], popt[2]), color='red')
			plt.legend([str(int(freq_arr[ii]))+' MHz'], fontsize=8)     
			plt.axis('off')
			plt.title("FWHM: %0.2fms" % (2.35*popt[2])) 
        plt.show()

        fig = plt.figure()
        sig_arr = np.array(sig_arr)
        plt.plot(freq_arr, sig_arr/sig_arr[len(sig_arr)//2], color='k', lw=3)
        plt.plot(freq_arr, (freq_arr/freq_arr.mean())**-3.)
        plt.plot(freq_arr, (freq_arr/freq_arr.mean())**-4.)
        plt.legend(['Data fit',r'$\propto\nu^{-3}$',r'$\propto\nu^{-4}$'])
        plt.xlim(1250, 1520)
        plt.xlabel('Freq (MHz)', fontsize=18)
        plt.ylabel('Pulse width (ms)', fontsize=18)
        plt.loglog()
        plt.show()

	return freq_arr, sig_arr

def fit_sim_frb(dm=1000.0, nfreq=1536, freq=(1219.70092773, 1519.50561523), 
                scat_tau_ref=0.001, spec_ind=0.0, dt=8.192e-5, width=0.0001, save_data=False):

        ntime = np.int(2*4148*dm*np.abs(freq[0]**-2-freq[-1]**-2)/dt)
        print(nfreq, ntime)
        undispersed_arrival_time = 0.5*ntime*dt
        undispersed_arrival_time -= 4148*dm*(max(freq)**-2)                                                              
        sp = simpulse.single_pulse(ntime, nfreq, min(freq), max(freq),
                           dm, scat_tau_ref, width, 10.0,
                           spec_ind, undispersed_arrival_time)

        data_simpulse = np.zeros([nfreq, ntime])
        sp.add_to_timestream(data_simpulse, 0.0, ntime*dt)
        data_event = data_simpulse[::-1]

        data_dedisp = tools.dedisperse(data_event, dm, freq=(freq[1], freq[0]))
#        plt.imshow(data_event, aspect='auto')
#        plt.show()

        if save_data:
                np.save('./herial.npy', data_dedisp)

	f,s = fit_width_freq(data_dedisp, freq=(freq[1], freq[0]), plot=True)
        s = np.array(s)
        
	plt.plot(f, s/s[len(s)//2], color='k', lw=3)
        plt.plot(f, (f/np.median(f))**-2.0)
#        plt.plot(f, (f/np.median(f))**-3.0)
        plt.plot(f, (f/np.median(f))**-4.0)
        plt.legend(['data','-2','-4'])
        plt.loglog()
	plt.show()

mm = np.argmax(arr[0].mean(0))
data = arr[0][:, mm-25:mm+25]
fit_width_freq(data, mk_plot=True)
#fit_sim_frb(dm=587.7, nfreq=1536*4, scat_tau_ref=0.01, freq=(1000,2000), width=0.000001, save_data=True)
