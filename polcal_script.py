from pol import *

generate_iquv_arr = True
bandpass_correct = True

dpath = '/tank/data/FRBs/FRB191108/iquv/CB21/numpyarr/stokes*sb37.npy'
dedisp_data_path = 'dedispersed_data'
bandpass_path = '/home/arts/connor/software/arts-analysis/arts-analysis/frb191108/bandpass_from_3c286_alpha-0.54_CB22.npy'

if generate_iquv_arr:
	arr_list, pulse_sample = make_iquv_arr(dpath, rebin_time=1, rebin_freq=1, dm=0.0, trans=True)
	stokes_arr = np.concatenate(arr_list)
	print(stokes_arr.shape)

if not generate_iquv_arr:
	try:
		stokes_arr = np.load(dedisp_data_path)
	except:
		print("No dedispersed Stokes array available. Exiting.")
		exit()

if bandpass_correct:
	bp_arr = np.load(bandpass_path)
	stokes_arr /= bp_arr[None, :, None]