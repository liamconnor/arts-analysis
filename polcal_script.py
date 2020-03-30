import os 

import numpy as np
import matplotlib.pylab as plt
import argparse 
import glob

import pol

freq_arr = pol.freq_arr
rebin_time = 1
rebin_freq = 1
pulse_width = 1 # number of samples to sum over
transpose = False

def generate_iquv_arr(dpath, dedisp_data_path=None, DM=0):
    if os.path.exists(dedisp_data_path):
        print("Reading %s in directly" % dedisp_data_path)
        stokes_arr = np.load(dedisp_data_path)
        pulse_sample = np.argmax(stokes_arr[0].mean(0))
    else:
        arr_list, pulse_sample = pol.make_iquv_arr(dpath, 
                                                   rebin_time=rebin_time, 
                                                   rebin_freq=rebin_freq, 
                                                   DM=DM, 
                                                   trans=False,
                                                   RFI_clean=True)
        stokes_arr = np.concatenate(arr_list, axis=0)
        stokes_arr = stokes_arr.reshape(4, pol.NFREQ//rebin_freq, -1)

        if type(dedisp_data_path)==str:
            stokes_arr_small = stokes_arr[:, :, 
                                pulse_sample-500:pulse_sample+500]
            np.save(dedisp_data_path, stokes_arr_small)

    return stokes_arr, pulse_sample

def read_dedisp_data(dpath):
    stokes_arr = np.load(dedisp_data_path)
    pulse_sample = np.argmax(stokes_arr[0].mean(0))

    return stokes_arr, pulse_sample

def plot_dedisp(stokes_arr, pulse_sample=None, pulse_width=1):
    #stokes_arr = stokes_arr[..., :len(stokes_arr[-1])//pulse_width*pulse_width]
    #stokes_arr = stokes_arr.reshape(4, -1, stokes_arr.shape[-1]//pulse_width, pulse_width).mean(-1)
    if pulse_sample is None:
        pulse_sample = np.argmax(stokes_arr[0].mean(0))
    
    pulse_sample /= pulse_width

    plt.subplot(211)
    plt.plot(stokes_arr[0].mean(0)-stokes_arr[0].mean())
    plt.plot(np.abs(stokes_arr[1]).mean(0)-np.abs(stokes_arr[1]).mean())
    plt.plot(np.abs(stokes_arr[2]).mean(0)-np.abs(stokes_arr[2]).mean())
    plt.plot(np.abs(stokes_arr[3]).mean(0)-np.abs(stokes_arr[3]).mean())
    plt.legend(['I', 'Q', 'U', 'V'])
    plt.subplot(212)
    plt.plot(stokes_arr[0].mean(0)-stokes_arr[0].mean())
    plt.plot(np.abs(stokes_arr[1]).mean(0)-np.abs(stokes_arr[1]).mean())
    plt.plot(np.abs(stokes_arr[2]).mean(0)-np.abs(stokes_arr[2]).mean())
    plt.plot(np.abs(stokes_arr[3]).mean(0)-np.abs(stokes_arr[3]).mean())
    plt.xlim(pulse_sample-100, pulse_sample+100)
    plt.xlabel('Sample number', fontsize=15)
    plt.show()

def bandpass_correct(stokes_arr, bandpass_path):
    bp_arr = np.load(bandpass_path)
    stokes_arr /= bp_arr[None, :, None]

    return stokes_arr

def xy_correct(stokes_arr, fn_xy_phase, plot=False, clean=False):
    stokes_arr_cal = np.zeros_like(stokes_arr)
    # Load xy phase cal from 3c286
    xy_phase = np.load(fn_xy_phase)
    use_ind_xy = np.arange(stokes_arr.shape[1])

    if clean:
        # abs_diff = np.abs(np.diff(xy_phase))
        # mu_xy = np.mean(abs_diff)
        # sig_xy = np.std(abs_diff)
        # mask_xy = list(np.where(abs_diff < (mu_xy+3*sig_xy))[0])
        mask_xy = range(235, 395)
        use_ind_xy = np.delete(use_ind_xy, mask_xy)

    xy_cal = np.poly1d(np.polyfit(freq_arr[use_ind_xy], 
                    xy_phase[use_ind_xy], 14))(freq_arr)
    for ii in range(4):
        stokes_arr[ii] -= np.median(stokes_arr[ii], keepdims=True, axis=-1)
    # Get FRB stokes I spectrum 
    I, Q, U, V = stokes_arr[0], stokes_arr[1], stokes_arr[2], stokes_arr[3]
    xy_data = U + 1j*V
    xy_data *= np.exp(-1j*xy_cal[:, None])
    stokes_arr_cal[2], stokes_arr_cal[3] = xy_data.real, xy_data.imag
    stokes_arr_cal[0] = stokes_arr[0]
    stokes_arr_cal[1] = stokes_arr[1]
    if plot:
        plt.plot(xy_phase)
        plt.plot(mask_xy, xy_phase[mask_xy])
        plt.plot(xy_cal, color='red')
        plt.legend(['XY_phase_calibrator', 'masked', 'Cal sol'])

    return stokes_arr_cal

def plot_xy_corr(data):
    if mk_plot and xy_correct:
        ext = [0, len(data[0,0])*1000/50.*pol.dt*1e3, 
               freq_arr.min(), freq_arr.max()]
        labels = ['Stokes I', 'Stokes Q', 'Stokes U', 'Stokes V']
        # Rebin in frequency and time
        Ispec = Ispec.reshape(-1, 16).mean(1)
        data = data[..., :data.shape[-1]//pulse_width*pulse_width]
        data = data.reshape(4, data.shape[1]//16, 16, 
                            data.shape[-1]//pulse_width, 
                            pulse_width).mean(2).mean(-1)
        for ii in range(4):
            plt.subplot(2,2,ii+1)
            data_med = np.median(data[ii],keepdims=True,axis=1)
            plt.imshow((data[ii]-data_med)/Ispec[:,None], 
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

def defaraday(data, pulse_sample=None, pulse_width=1):
    """ Pulse sample should be None, a slice, or an integer.
    """ 
    if pulse_sample is None:
        pulse_sample = np.argmax(data[0].mean(0))

    Ispec = data[0, :, pulse_sample]
    Q = (data[1]-np.median(data[1],keepdims=True,axis=1))/Ispec[:,None]
    U = (data[2]-np.median(data[2],keepdims=True,axis=1))/Ispec[:,None]
    V = (data[3]-np.median(data[3],keepdims=True,axis=1))/Ispec[:,None]
    Q = Q[:, pulse_sample//pulse_width]
    U = U[:, pulse_sample//pulse_width]

    results_faraday = pol.derotate_faraday(Q, U, pulse_sample=None, 
                                           pulse_width=1, RMmin=-1e4, 
                                           RMmax=1e4)
    Qcal, Ucal, P_cal, rm_bf, lam_arr, phase_std, P = results_faraday
    plt.plot(np.angle(P_cal))
    plt.show()

def mk_plot(stokes_arr, pulse_sample=None):
    if pulse_sample is None:
        pulse_sample = np.argmax(stokes_arr[0].mean(0))
    pol.plot_im_raw(stokes_arr, pulse_sample=pulse_sample)
    pol.plot_raw_data(stokes_arr, pulse_sample=pulse_sample)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs polarisation pipeline",
                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--basedir', 
                        help='base directory of polarisation data', 
                        type=str, required=True)
    parser.add_argument('-A', '--All', 
                        help='Do every step in the pipeline', action='store_true')
    parser.add_argument('-p', '--polcal', 
                        help='generate iquv array', action='store_true')
    parser.add_argument('-g', '--gen_arr', 
                        help='generate iquv array', action='store_true')
    parser.add_argument('-F', '--faraday', 
                        help='Faraday fit and de-rotate', 
                        action='store_true')
    parser.add_argument('-sb', '--gen_sb', 
                        help='generate SB from npy files', 
                        action='store_true')
    parser.add_argument('-pd', '--plot_dedisp', 
                        help='plot 1D stokes data in time', 
                        action='store_true')
    parser.add_argument('-ps', '--plot_stokes', 
                        help='plot 2D stokes data', 
                        action='store_true')
    parser.add_argument('-c', '--calibrate_frb', 
                        help='use non-switch polcal solution to cal FRB', 
                        action='store_true')
    parser.add_argument('-b', '--bandpass_file', help='correct bandpass', 
                        default=None, type=str)
    parser.add_argument('-pw', '--pulse_width', help='', 
                        default=1, type=int)
    parser.add_argument('-src', '--src', help='source name', 
                        default='3C286', type=str)
    parser.add_argument('-rmmin', '--rmmin', help='min RM to search', 
                        default=-1e4, type=float)
    parser.add_argument('-rmmax', '--rmmax', help='max RM to search', 
                        default=1e4, type=float)

    inputs = parser.parse_args()
    obs_name = inputs.basedir.split('/')[4]
    try:
        params = glob.glob(inputs.basedir+'/numpyarr/DM*txt')[0]
    except:
        print("Expected a txt file with DM, width, CB, and SB in %s" % 
            (inputs.basedir+'/numpyarr/'))
        print("e.g. numpyarr/DM588.13_SNR60_CB21_SB37_Width5.txt")
        exit()

    DM = float(params.split('DM')[-1].split('_')[0])
    SB = int(params.split('SB')[-1].split('_')[0])

    if inputs.gen_sb or inputs.All:
        print("Generating SB from npy data")

        if inputs.polcal or inputs.All:
            folder_polcal = inputs.basedir+'/polcal/'
            pol.sb_from_npy(folder_polcal, sb=35, off_src=False)
            pol.sb_from_npy(folder_polcal, sb=35, off_src=True)

        folder = inputs.basedir+'/numpyarr/'
        pol.sb_from_npy(folder, sb=SB, off_src=False)


    if inputs.polcal or inputs.All:
        print("Getting bandpass and xy pol solution from %s" % inputs.src)
        stokes_arr_spec, bandpass, xy_phase = pol.calibrate_nonswitch(
                                                        inputs.basedir, 
                                                        src=inputs.src, 
                                                        save_sol=True)

    if inputs.gen_arr or inputs.All:
        print("Assuming %0.2f for %s" % (DM, obs_name))
        dpath = inputs.basedir + '/numpyarr/stokes*sb*.npy'
        dedisp_data_path = inputs.basedir+'/numpyarr/%s_dedisp.npy' % obs_name
        stokes_arr, pulse_sample = generate_iquv_arr(dpath, 
                                    dedisp_data_path=dedisp_data_path, DM=DM)

    if inputs.plot_dedisp or inputs.All:
        try:
           stokes_arr
        except NameError:
           print("Cannot plot data if there is no stokes array")
           exit()
        plot_dedisp(stokes_arr, pulse_sample=pulse_sample, 
                    pulse_width=inputs.pulse_width)

    if inputs.calibrate_frb or inputs.All:
        try:
           stokes_arr
        except NameError:
           print("Cannot calibrate FRB if there is no stokes array")
           exit()

        fn_bandpass = inputs.basedir+'/polcal/bandpass.npy'
        fn_xy_phase = inputs.basedir+'/polcal/xy_phase.npy'
        print("Calibrating bandpass")
        stokes_arr_cal = bandpass_correct(stokes_arr, fn_bandpass)
        print("Calibrating xy correlation with %s" % inputs.src)
        stokes_arr_cal = xy_correct(stokes_arr_cal, fn_xy_phase, 
                                    plot=True, clean=True)

    if inputs.plot_stokes or inputs.All:
        mk_plot(stokes_arr.reshape(4, 1536//16, 16, -1).mean(-2), 
                pulse_sample=pulse_sample)
        try:
           stokes_arr_cal
           mk_plot(stokes_arr_cal.reshape(4, 1536//16, 16, -1).mean(-2), 
                   pulse_sample=pulse_sample)
        except NameError:
           print("Cannot plot calibrated data if there is no stokes_arr_cal")

    if inputs.faraday or inputs.All:
        print("Faraday fitting between %0.1f and %0.1f rad m**-2" % 
                    (inputs.rmmin, inputs.rmmax))

        try:
            stokes_vec = stokes_arr_cal[..., pulse_sample-1:pulse_sample+1].mean(-1)            
        except:
            print("Using uncalibrated data")
            stokes_vec = stokes_arr[..., pulse_sample-1:pulse_sample+1].mean(-1)            

        results_faraday = pol.faraday_fit(stokes_vec, RMmin=inputs.rmmin, 
                                   RMmax=inputs.rmmax, nrm=1000, nphi=200)
        RMs, P_derot_arr, RMmax, phimax, derot_phase = results_faraday
        print(RMmax, phimax)
        fig=plt.figure()
        plt.plot(RMs, np.max(P_derot_arr, axis=-1))
        fig=plt.figure()
        extent = [inputs.rmmin, inputs.rmmax, 0, 360]
        plt.imshow(P_derot_arr, aspect='auto', vmax=P_derot_arr.max(), 
                   vmin=P_derot_arr.max()*0.5, extent=extent)
        plt.xlabel('Phi (deg)', fontsize=16)
        plt.ylabel('RM (rad m**-2)', fontsize=16)
        plt.show()







