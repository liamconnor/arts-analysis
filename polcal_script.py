import numpy as np
import matplotlib.pylab as plt

import pol

generate_iquv_arr = False
plot_dedisp = True
bandpass_correct = True
RFI_clean = True
mk_plot = True
xy_correct = True
defaraday = True

freq_arr = pol.freq_arr
nfreq = 1536
rebin_time = 1
rebin_freq = 1
DM = 1290.0
DM = 348.5
DM = 832.0
dt = 8.192e-5
pulse_width = 25 # number of samples to sum over
transpose = False

dpath = '/tank/data/FRBs/FRB200322/iquv/numpyarr/stokes*sb29*.npy'
dpath = '/tank/data/FRBs/R3/20200322/iquv/numpyarr/stokes*sb35*.npy'
dpath = '/tank/data/FRBs/FRB200323/iquv/numpyarr/stokes*sb18*.npy'
dedisp_data_path = '/tank/data/FRBs/FRB200323/iquv/numpyarr/FRB200323_dedisp.npy'
bandpass_path = '/tank/data/FRBs/FRB200216/iquv/3C286/CB05/on/npy/stokesbandpass_from_3c286_alpha-0.54_CB05.npy'
xy_phase_cal = '/tank/data/FRBs/FRB200216/iquv/3C286/CB05/on/npy/stokesxy_phase_3c286_frequency.npy'

if generate_iquv_arr:
    arr_list, pulse_sample = pol.make_iquv_arr(dpath, rebin_time=rebin_time, 
                                               rebin_freq=rebin_freq, dm=DM, trans=transpose,
                                               RFI_clean=RFI_clean)
    stokes_arr = np.concatenate(arr_list, axis=0)
    stokes_arr = stokes_arr.reshape(4, nfreq//rebin_freq, -1)
    np.save(dedisp_data_path,stokes_arr[:, :, pulse_sample-500:pulse_sample+500])

if not generate_iquv_arr:
    try:
        stokes_arr = np.load(dedisp_data_path)
        pulse_sample = np.argmax(stokes_arr[0].mean(0))
    except:
        print("No dedispersed Stokes array available. Exiting.")
        exit()

if plot_dedisp:
    plt.plot(stokes_arr[0].mean(0)-stokes_arr[0].mean())
    plt.plot(np.abs(stokes_arr[1]).mean(0)-np.abs(stokes_arr[1]).mean())
    plt.plot(np.abs(stokes_arr[2]).mean(0)-np.abs(stokes_arr[2]).mean())
    plt.plot(np.abs(stokes_arr[3]).mean(0)-np.abs(stokes_arr[3]).mean())
    plt.axvline(pulse_sample, color='k', linestyle='--', alpha=0.25)
    plt.show()

if bandpass_correct:
    bp_arr = np.load(bandpass_path)
    stokes_arr /= bp_arr[None, rebin_freq//2::rebin_freq, None]

if xy_correct:
    # Reverse frequency order
    data = stokes_arr[:, ::-1]
    # Load xy phase cal from 3c286
    xy_phase = np.load(xy_phase_cal)
    xy_cal = np.poly1d(np.polyfit(freq_arr, xy_phase, 7))(freq_arr)
    # Get FRB stokes I spectrum 
    Ispec = data[0,:,pulse_sample-pulse_width//2:pulse_sample+pulse_width//2].mean(-1)
    I, Q, U, V = data[0,:,:], data[1,:,:], data[2,:,:], data[3,:]
    xy_data = data[2] + 1j*data[3]
    xy_data *= np.exp(-1j*xy_cal[:, None])
    data[2], data[3] = xy_data.real, xy_data.imag

if mk_plot and xy_correct:
    ext = [0, len(data[0,0])*1000/50.*dt*1e3, freq_arr.min(), freq_arr.max()]
    labels = ['Stokes I', 'Stokes Q', 'Stokes U', 'Stokes V']
    # Rebin in frequency and time
    Ispec = Ispec.reshape(-1, 16).mean(1)
    data = data[..., :data.shape[-1]//pulse_width*pulse_width]
    data = data.reshape(4, data.shape[1]//16, 16, data.shape[-1]//pulse_width, pulse_width).mean(2).mean(-1)
    for ii in range(4):
        plt.subplot(2,2,ii+1)
        plt.imshow((data[ii]-np.median(data[ii],keepdims=True,axis=1))/Ispec[:,None], 
                   aspect='auto', extent=ext)
        plt.text(50, 1480, labels[ii], color='white', fontsize=12)
        if ii%2==0:
            plt.ylabel('Freq (MHz)')
        plt.yticks([1500, 1400, 1300])  
        if ii>1:
            plt.xlabel('Time (ms)')
        plt.xlim(600, 1000)
    plt.show()
    exit()

if defaraday:
    ii=1;Q = (data[ii]-np.median(data[ii],keepdims=True,axis=1))/Ispec[:,None]
    ii=2;U = (data[ii]-np.median(data[ii],keepdims=True,axis=1))/Ispec[:,None]
    ii=3;V = (data[ii]-np.median(data[ii],keepdims=True,axis=1))/Ispec[:,None]
    Q = Q[:, pulse_sample//pulse_width]
    U = U[:, pulse_sample//pulse_width]
    Qcal, Ucal, P_cal, rm_bf, lam_arr, phase_std, P = pol.derotate_faraday(Q, U, pulse_sample=None, pulse_width=1, RMmin=-1e4, RMmax=1e4)
    plt.plot(np.angle(P_cal))
    plt.show()

if mk_plot:
    pol.plot_im_raw(stokes_arr, pulse_sample=pulse_sample)
    pol.plot_raw_data(stokes_arr, pulse_sample=pulse_sample)
