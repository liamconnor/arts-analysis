import numpy as np
import matplotlib.pylab as plt
import glob 
import matplotlib.pylab as plt

import tools

dpath='/tank/users/oostrum/iquv/FRBs/20191108-18:11:00_FRBfield/CB21/stokes*_sb37.npy'
stokes_ps = ['I', 'Q', 'U', 'V']
DM = 588.

NFREQ = 1536
freq = (1219.70092773,1519.50561523)
freq_arr = np.linspace(freq[0], freq[-1], NFREQ)
lam_arr = 3e2 / freq_arr
dt = 8.192e-5
rebin_time=1
rebin_freq=1

def make_iquv_arr(dpath, rebin_time=1, rebin_freq=1):
	""" Read in all 4 arrays, dedisperse, 
	return list with median-subtracted, rebinned 
	[arr_I, arr_Q, arr_U, arr_V]

	Parameters
	----------
	dpath : str 
		data path with .npy IQUV files
	"""
	flist = glob.glob(dpath)
	flist.sort() # should be IQUV ordered now
	arr_list = []

	for ii, fn in enumerate(flist):
		print("Assuming %s is Stokes %s" % (fn, stokes_ps[ii]))
		arr = np.load(fn)
		last_ind = -int(abs(4150*DM*(freq[0]**-2-freq[-1]**-2)/dt))
		arr = tools.dedisperse(arr.transpose(), DM, freq=freq)[:, :last_ind]
		nt, nf = arr.shape[-1], arr.shape[0]
		arr = arr - np.median(arr, axis=-1, keepdims=True)
		arr = arr[:nf//rebin_freq*rebin_freq, :nt//rebin_time*rebin_time]
		arr = arr.reshape(nf//rebin_freq, rebin_freq, nt//rebin_time, rebin_time).mean(1).mean(-1)
		arr_list.append(arr)

	pulse_sample = np.argmax(arr_list[0].mean(0))

	return arr_list, pulse_sample

def derotate_UV(arr_U, arr_V, pulse_sample=None, pulse_width=1):
	""" Create complex xy spectrum from U/V. Find phase 
	such that flucations in V are minimized. Derotate xy 
	and return calibrated U/V. 

	Parameters:
	-----------
	arr_U : array
		(nfreq, ntime)
	arr_V : array 
		(nfreq, ntime)
	pulse_sample: int 
		time sample with pulse 
	pulse_width : int 
		number of samples to average over in time

	Returns:
	--------
	Ucal, Vcal, xy_cal, phi_bf
	"""
	xy = arr_U + 1j*arr_V

	if pulse_sample is None:
		pulse_sample = np.argmax(np.sqrt(arr_U**2 + arr_V**2).mean(0))

	if pulse_width>1:
		xy_pulse = xy[:, pulse_sample-pulse_width//2:pulse_sample+pulse_width//2].mean(-1)
	else:
		xy_pulse = xy[:, pulse_sample]

	phis = np.linspace(-np.pi, np.pi, 1000)

	phase_std = []
	for phi in phis:
		xy_rot = xy_pulse * np.exp(-1j*phi)
		phase_std.append(np.std(xy_rot.imag))

	# Assume best fit phase is the one that minimizes 
	# oscillation in V
	phi_bf = phis[np.argmin(np.array(phase_std))]
	xy_cal = xy_pulse*np.exp(-1j*phi_bf)
	Ucal, Vcal = xy_cal.real, xy_cal.imag

	return Ucal, Vcal, xy_cal, phi_bf

def derotate_faraday(arr_Q, arr_U, pulse_sample=None, pulse_width=1, RMmin=0.0, RMmax=1e4):
	""" Create complex xy spectrum from U/V. Find phase 
	such that flucations in V are minimized. Derotate xy 
	and return calibrated U/V. 
	"""
	arr_Q = arr_Q[:, pulse_sample-100:pulse_sample+100]	
	arr_U = arr_U[:, pulse_sample-100:pulse_sample+100]
	arr_Q -= np.median(arr_Q, axis=-1, keepdims=True)
	arr_U -= np.median(arr_U, axis=-1, keepdims=True)

	P = arr_Q + 1j*arr_U
	lam_arr = 3e2 / np.linspace(freq[0], freq[-1], P.shape[0])

	if pulse_sample is None:
		pulse_sample = np.argmax(np.sqrt(arr_Q**2 + arr_U**2).mean(0))

	if pulse_width>1:
		P_pulse = P[:, 100-pulse_width//2:100+pulse_width//2].mean(-1)
	else:
		P_pulse = P[:, 100]

	phis = np.linspace(-2*np.pi, 2*np.pi, 1000)
	RMs = np.linspace(-RMmax, RMmax, 50000)

	phase_std = []
	for rm in RMs:
		P_rot = P_pulse * np.exp(-2j*rm*lam_arr**2)
		#phase_std.append(np.std(np.angle(P_rot)))
		phase_std.append(P_rot.real.sum())

	# Assume best fit phase is the one that minimizes 
	# oscillation in V
	rm_bf = RMs[np.argmin(np.array(phase_std))]
	P_cal = P_pulse*np.exp(-2j*rm_bf*lam_arr**2)
	Qcal, Ucal = P_cal.real, P_cal.imag

	return Qcal, Ucal, P_cal, rm_bf, lam_arr, phase_std, P

def plot_raw_data(arr, pulse_sample=None):
	""" Plot data before calibration
	"""
	fig = plt.figure(figsize=(7,7))

	if pulse_sample is None:
		pulse_sample = np.argmax(arr[0].mean(0))
	
	freq_arr = np.linspace(freq[0], freq[-1], arr[0].shape[0])
	plt.subplot(211)
	[plot(freq_arr, arr[jj][:, samp], alpha=0.7, lw=3) for jj in range(4)]
	plt.legend(['uncal I','uncal Q','uncal U','uncal V'])

	plt.subplot(212)
	plt.plot(freq_arr, (np.sqrt(arr[3][:, pulse_sample]**2 + arr[1][:, pulse_sample]**2 \
					+ arr[2][:, pulse_sample]**2)/arr[0][:, pulse_sample]), 
					color='k', alpha=0.9)
	plt.ylim(0,1)
	plt.xlabel('Freq', fontsize=18)
	plt.ylabel('Pol fraction [uncal]', fontsize=18)

	plt.show()

def plot_im_raw(arr, pulse_sample=None):
	if pulse_sample is None:
		pulse_sample = np.argmax(arr[0].mean(0))

	fig = plt.figure(figsize=(7,7))
	plt.subplot(221)
	plt.imshow(arr[0][:, pulse_sample-50:pulse_sample+50], aspect='auto')
	plt.subplot(222)
	plt.imshow(arr[0][:, pulse_sample-50:pulse_sample+50], aspect='auto')
	plt.subplot(223)
	plt.imshow(arr[0][:, pulse_sample-50:pulse_sample+50], aspect='auto')
	plt.subplot(224)
	plt.imshow(arr[0][:, pulse_sample-50:pulse_sample+50], aspect='auto')
	plt.show()


# phi_bestfit, RMbestfit = -1.2755265051007838, 472.54725472547256
# phase_xy_bestfit = 2.05

# NFREQ = 1536
# freq = (1219.70092773,1519.50561523)
# freq_arr = np.linspace(freq[0], freq[-1], NFREQ)
# lam_arr = 3e2 / freq_arr

# mask = np.loadtxt('/home/arts/.controller/amber_conf/zapped_channels_1370.conf')

# fnI = directory + 'stokesI_sb37.npy'
# fnQ = directory + 'stokesQ_sb37.npy'
# fnU = directory + 'stokesU_sb37.npy'
# fnV = directory + 'stokesV_sb37.npy'

# #fnI = '/tank/users/oostrum/iquv/B0531/2019-10-17-04:29:24.B0531+21/stokesI.npy'
# #fnQ = '/tank/users/oostrum/iquv/B0531/2019-10-17-04:29:24.B0531+21/stokesQ.npy'
# #fnU = '/tank/users/oostrum/iquv/B0531/2019-10-17-04:29:24.B0531+21/stokesU.npy'
# #fnV = '/tank/users/oostrum/iquv/B0531/2019-10-17-04:29:24.B0531+21/stokesV.npy'

# #fn = '/tank/users/oostrum/iquv/B1404+286/B1404off_'
# #fnI = fn + 'I.npy'
# #fnQ = fn + 'Q.npy'
# #fnU = fn + 'U.npy'
# #fnV = fn + 'V.npy'

# I = np.load(fnI)
# Q = np.load(fnQ)
# U = np.load(fnU)
# V = np.load(fnV)

# dI = tools.dedisperse(I.transpose(), DM, freq=freq)[:, :-2500]
# dQ = tools.dedisperse(Q.transpose(), DM, freq=freq)[:, :-2500]
# dU = tools.dedisperse(U.transpose(), DM, freq=freq)[:, :-2500]
# dV = tools.dedisperse(V.transpose(), DM, freq=freq)[:, :-2500]

# dI = dI - np.median(dI, axis=-1, keepdims=True)
# dQ = dQ - np.median(dQ, axis=-1, keepdims=True)
# dU = dU - np.median(dU, axis=-1, keepdims=True)
# dV = dV - np.median(dV, axis=-1, keepdims=True)

# dI = dI.reshape(1536, -1, 1).mean(-1)
# dQ = dQ.reshape(1536, -1, 1).mean(-1)
# dU = dU.reshape(1536, -1, 1).mean(-1)
# dV = dV.reshape(1536, -1, 1).mean(-1)

# mm = np.argmax(dI.mean(0))

# dI = dI[:, mm-100:mm+100]
# dQ = dQ[:, mm-100:mm+100]
# dU = dU[:, mm-100:mm+100]
# dV = dV[:, mm-100:mm+100]

# xy = dU + 1j*dV
# xy *= np.exp(-1j*phase_xy_bestfit)
# dU = xy.real
# dV = xy.imag

# P = dQ[:,mm] + 1j*dU[:,mm]
# P = P.reshape(-1, 16).mean(-1)

# thetas = np.linspace(0, 4*np.pi, 1000)
# RMs = np.linspace(100, 1000, 10000)

# lam_arr = lam_arr[::16]
# S=[]
# for rm in RMs:
#     P_ = P*np.exp(-2j*rm*lam_arr**2)
#     S.append(np.std(np.angle(P_)))

# S = np.array(S)
# np.save('Pvec.npy', P)
# rmbf = RMs[np.argmin(S)]
# Pcal = P * np.exp(-2j*lam_arr**2*rmbf)
# phi = np.mean(np.angle(Pcal))
# print(phi, rmbf)
# Pcal *= np.exp(-1j*phi)
# derotate = np.exp(-2j*lam_arr**2*RMbestfit - 1j*phi_bestfit)

# phi_bestfit, RMbestfit = -1.2755265051007838, 472.54725472547256
# phase_xy_bestfit = 2.05

# plt.plot(lam_arr, Pcal.real)
# plt.plot(lam_arr, Pcal.imag)
# plt.show()
# exit()
# plt.plot(dU[:,mm])
# plt.plot(P)
# plt.plot(P.imag)
# plt.show()
# exit()

# plt.subplot(221)
# plt.imshow(dI[:, mm-50:mm+50], aspect='auto')
# plt.subplot(222)
# plt.imshow(dQ[:, mm-50:mm+50], aspect='auto')
# plt.subplot(223)
# plt.imshow(dU[:, mm-50:mm+50], aspect='auto')
# plt.subplot(224)
# plt.imshow(dV[:, mm-50:mm+50], aspect='auto')
# plt.show()
# exit()


# std_I = np.std(dI, axis=-1)[:, None]
# dI /= std_I
# dQ /= std_I
# dU /= std_I
# dV /= std_I

# dI[mask.astype(int)] = 0.0
# dQ[mask.astype(int)] = 0.0
# dU[mask.astype(int)] = 0.0
# dV[mask.astype(int)] = 0.0

# indmax = np.argmax(dI.mean(0))

# #imshow(dI[:, indmax-50:indmax+50], aspect='auto', vmax=3, vmin=-2)

# mu = (I - Q) / (I + Q)
