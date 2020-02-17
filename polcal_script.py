import numpy as np

import pol

generate_iquv_arr = True
bandpass_correct = True
RFI_clean = True
mk_plot = True

freq_arr = pol.freq_arr
nfreq = 1536
rebin_time = 1
rebin_freq = 1
DM = 478.74
pulse_width = 10 # number of samples to sum over
transpose = False

dpath = '/tank/data/FRBs/FRB200216/iquv/CB05/npy/stokes*sb30.npy'
dedisp_data_path = '/home/arts/connor/software/arts-analysis/arts-analysis/frb191108/dedispersed_FRB191108.npy'
bandpass_path = '/tank/data/FRBs/FRB200216/iquv/3C286/CB05/on/npy/stokesbandpass_from_3c286_alpha-0.54_CB05.npy'
xy_phase_cal = '/tank/data/FRBs/FRB200216/iquv/3C286/CB05/on/npy/stokesxy_phase_3c286_frequency.npy'

if generate_iquv_arr:
	arr_list, pulse_sample = pol.make_iquv_arr(dpath, rebin_time=rebin_time, 
											   rebin_freq=rebin_freq, dm=DM, trans=transpose,
											   RFI_clean=RFI_clean)
	stokes_arr = np.concatenate(arr_list, axis=0).reshape(4, nfreq, -1)

if not generate_iquv_arr:
	try:
		stokes_arr = np.load(dedisp_data_path)
		pulse_sample = np.argmax(stokes_arr[0].mean(0))
	except:
		print("No dedispersed Stokes array available. Exiting.")
		exit()

if bandpass_correct:
	bp_arr = np.load(bandpass_path)
	stokes_arr /= bp_arr[None, :, None]

if xy_correct:
	xy_phase = np.load(xy_phase_cal)
    xy_phase[200:400] = 2.6
    xy_cal = np.poly1d(np.polyfit(freq_arr, xy_phase, 7))(freq_arr)
    plt.plot(freq_arr, xy_phase)
    plt.plot(freq_arr, xy)
    plt.show()

if mk_plot:
	pol.plot_im_raw(stokes_arr, pulse_sample=pulse_sample)
	pol.plot_raw_data(stokes_arr, pulse_sample=pulse_sample)