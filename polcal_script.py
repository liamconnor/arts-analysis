from pol import *

generate_iquv_arr = True
bandpass_correct = True
mk_plot = True

nfreq = 1536
rebin_time = 1
rebin_freq = 1
DM = 588.1
pulse_width = 10 # number of samples to sum over
transpose = False

dpath = '/tank/data/FRBs/FRB200216/iquv/CB05/npy/stokes*sb30.npy'
dedisp_data_path = '/home/arts/connor/software/arts-analysis/arts-analysis/frb191108/dedispersed_FRB191108.npy'
bandpass_path = '/home/arts/connor/software/arts-analysis/arts-analysis/frb191108/bandpass_from_3c286_alpha-0.54_CB22.npy'

if generate_iquv_arr:
	arr_list, pulse_sample = make_iquv_arr(dpath, rebin_time=rebin_time, rebin_freq=rebin_freq, dm=DM, trans=transpose)
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

if mk_plot:
	plot_im_raw(stokes_arr, pulse_sample=pulse_sample)
	plot_raw_data(stokes_arr, pulse_sample=pulse_sample)