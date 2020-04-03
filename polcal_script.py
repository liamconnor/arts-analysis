import os

import numpy as np
import matplotlib.pylab as plt
import argparse 
import glob

import tools
import pol

freq_arr = pol.freq_arr
rebin_time = 1
rebin_freq = 1
pulse_width = 1 # number of samples to sum over
transpose = False
SNRtools = tools.SNR_Tools()

def generate_iquv_arr(dpath, dedisp_data_path=None, DM=0):
    print(dedisp_data_path)
    if os.path.exists(dedisp_data_path):
        print("Reading %s in directly" % dedisp_data_path)
        stokes_arr = np.load(dedisp_data_path)
        pulse_sample = np.argmax(stokes_arr[0].mean(0))
        stokes_arr = stokes_arr[..., pulse_sample-500:pulse_sample+500]
        pulse_sample = 500
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

def get_width(data):
    snr_max, width_max = SNRtools.calc_snr_matchedfilter(data, widths=range(500))
    return snr_max, width_max

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

def check_dirs(fdir):
    """ Expecting: 
    fdir/numpyarr
    fdir/polcal
    fdir/dada
    fdir/numpyarr/
    """
    if not os.path.isdir(fdir+'/numpyarr'):
        os.mkdir(fdir+'/numpyarr')

    if not os.path.isdir(fdir+'/polcal'):
        print('Making empty polcal dir')
        os.mkdir(fdir+'/polcal')

    if not os.path.isdir(fdir+'/dada'):
        print('Making empty dada dir')
        os.mkdir(fdir+'/dada/')        

def plot_RMspectrum(RMs, P_derot_arr, RMmax, 
                    phimax, derot_phase, fn_fig=None):
    fig=plt.figure()
    plt.plot(RMs, np.max(P_derot_arr, axis=-1))
    plt.ylabel('Defaraday amplitude', fontsize=16)
    plt.xlabel(r'RM (rad m$^{-2}$)', fontsize=16)
    plt.axvline(RMmax, color='r', linestyle='--')
    plt.text(RMmax*0.5, np.max(P_derot_arr)*0.8, '%s \n RMmax~%d' % (obs_name, RMmax))
    fig=plt.figure()
    extent=[0, 360, inputs.rmmin, inputs.rmmax]
    plt.imshow(P_derot_arr, aspect='auto', vmax=P_derot_arr.max(), 
              vmin=P_derot_arr.max()*0.5, extent=extent)
    plt.xlabel('Phi (deg)', fontsize=16)
    plt.ylabel('RM (rad m**-2)', fontsize=16)

    if fn_fig is not None:
        plt.savefig(fn_fig)

def plot_all(stoke_arr, suptitle='', fds=16):
    stokes_arr_ = stokes_arr.reshape(4, 1536//fds, fds, -1).mean(2)
    stokes_arr_ = stokes_arr_[..., :stokes_arr.shape[-1]//5*5].reshape(4,96,-1,5).mean(-1)
    plt.subplot(421)
    plt.plot(stokes_arr_[0].mean(0))
    plt.ylabel('I')
    plt.subplot(422)
    plt.imshow(stokes_arr_[0]-stokes_arr_[0].mean(-1)[:, None], aspect='auto')
    plt.subplot(423)
    plt.plot(stokes_arr_[1].mean(0))
    plt.ylabel('Q')
    plt.subplot(424)
    plt.imshow(stokes_arr_[1]-stokes_arr_[1].mean(-1)[:, None], aspect='auto')
    plt.subplot(425)
    plt.plot(stokes_arr_[2].mean(0))
    plt.ylabel('U')
    plt.subplot(426)
    plt.imshow(stokes_arr_[2]-stokes_arr_[2].mean(-1)[:, None], aspect='auto')
    plt.subplot(427)
    plt.plot(stokes_arr_[3].mean(0))
    plt.ylabel('V')
    plt.subplot(428)
    plt.imshow(stokes_arr_[3]-stokes_arr_[3].mean(-1)[:, None], aspect='auto')
    plt.xlabel('Time (samples)')
    plt.suptitle(suptitle)
    plt.plot()


def mk_pol_plot(stokes_arr, pulse_sample=None, pulse_width=1):
    if pulse_sample is None:
        pulse_sample = np.argmax(stokes_arr[0].mean(0))
    pol.plot_im_raw(stokes_arr, pulse_sample=pulse_sample, pulse_width=pulse_width)
    pol.plot_raw_data(stokes_arr, pulse_sample=pulse_sample, pulse_width=pulse_width)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs polarisation pipeline",
                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--basedir', 
                        help='base directory of polarisation data', 
                        type=str, required=True)
    parser.add_argument('-A', '--All', 
                        help='Do every step in the pipeline', action='store_true')
    parser.add_argument('-dd', '--dada', 
                        help='generate numpy files from dada',
                        action='store_true')
    parser.add_argument('-sb', '--gen_sb', 
                        help='generate SB from npy files', 
                        action='store_true')
    parser.add_argument('-g', '--gen_arr', 
                        help='generate iquv array', action='store_true')
    parser.add_argument('-pc', '--polcal', 
                        help='generate iquv array', action='store_true')
    parser.add_argument('-c', '--calibrate_frb', 
                        help='use non-switch polcal solution to cal FRB', 
                        action='store_true')
    parser.add_argument('-F', '--faraday', 
                        help='Faraday fit and de-rotate', 
                        action='store_true')
    parser.add_argument('-p', '--mk_plot', 
                        help='plot 1D stokes data in time', 
                        action='store_true')
    parser.add_argument('-ps', '--plot_stokes', 
                        help='plot 2D stokes data', 
                        action='store_true')
    parser.add_argument('-src', '--src', help='calibrator source name', 
                        default='3C286', type=str)
    parser.add_argument('-rmmin', '--rmmin', help='min RM to search', 
                        default=-1e4, type=float)
    parser.add_argument('-rmmax', '--rmmax', help='max RM to search', 
                        default=1e4, type=float)
    parser.add_argument('-fds', '--freq_downsample', help='downsample in freq', 
                        default=16, type=int)

    
    inputs = parser.parse_args()
    obs_name = inputs.basedir.split('/')[4]

    check_dirs(inputs.basedir)

    if inputs.dada or inputs.All:
        fndada = glob.glob(inputs.basedir+'/dada/*dada')[0]
        outdir = inputs.basedir+'/numpyarr/'
        print("Converting dada into numpy for %s" % fndada)
#        os.system('./read_IQUV_dada.py %s --outdir %s' % (fndada, outdir))
        if inputs.polcal:
            fndada = glob.glob(inputs.basedir+'/polcal/*dada')[0]
            outdir = inputs.basedir+'/polcal/'
            os.system('./read_IQUV_dada.py %s --outdir %s' % (fndada, outdir))

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
        flist_sb = glob.glob(dpath)
        if len(flist_sb)==-1:
            print("No SB data, cannot generate stokes array")
        else:
            dedisp_data_path = inputs.basedir+'/numpyarr/%s_dedisp.npy' % obs_name

            if not os.path.exists(dedisp_data_path):
                fn_dedisp = inputs.basedir+'/numpyarr/*_dedisp.npy'
                dedisp_data_path = glob.glob(fn_dedisp)[0]

            stokes_arr, pulse_sample = generate_iquv_arr(dpath, 
                                    dedisp_data_path=dedisp_data_path, DM=DM)

            snr_max, width_max = get_width(stokes_arr[0].mean(0))

    try:
        stokes_arr
    except:
        print('stokes_arr not defined')
        exit()

    if inputs.mk_plot or inputs.All:
        try:
           stokes_arr
        except NameError:
           print("Cannot plot data if there is no stokes array")
           exit()
        plot_dedisp(stokes_arr, pulse_sample=pulse_sample, 
                    pulse_width=width_max)

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
                                    plot=inputs.mk_plot, clean=True)

    if inputs.mk_plot:
        plot_all(stokes_arr, suptitle='Uncalibrated', 
                 fds=inputs.freq_downsample)
        # mk_pol_plot(stokes_arr.reshape(4, 1536//16, 16, -1).mean(2),
        #         pulse_sample=pulse_sample, pulse_width=8)
        try:
           stokes_arr_cal
           plot_all(stokes_arr_cal, suptitle='xy-Calibrated', 
                    fds=inputs.freq_downsample)
           # mk_pol_plot(stokes_arr_cal.reshape(4, 1536//1, 1, -1).mean(-2), 
           #         pulse_sample=pulse_sample, pulse_width=8)
        except NameError:
           print("Cannot plot calibrated data if there is no stokes_arr_cal")

    if inputs.faraday or inputs.All:
        print("Faraday fitting between %0.1f and %0.1f rad m**-2" % 
                    (inputs.rmmin, inputs.rmmax))

        try:
            stokes_vec = stokes_arr_cal
        except:
            print('Using uncalibrated data')
            stokes_vec = stokes_arr

        ntime = stokes_vec.shape[-1]
        stokes_vec = stokes_vec[..., :ntime//width_max*width_max]
        stokes_vec = stokes_vec.reshape(4, 1536, -1, width_max).mean(-1)
        pulse_sample = np.argmax(stokes_vec[0].mean(0))
        stokes_vec = stokes_vec[..., pulse_sample]

        if inputs.mk_plot:
            plot_all(stokes_arr, suptitle='Uncalibrated', 
                     fds=inputs.freq_downsample)

        if not inputs.polcal:
            Ucal, Vcal, xy_cal, phi_xy = pol.derotate_UV(stokes_vec[2], stokes_vec[3])
            stokes_vec[2] = Ucal
            stokes_vec[3] = Vcal
            U,V = stokes_arr[2], stokes_arr[3]
            XY = U+1j*V
            XY *= np.exp(-1j*phi_xy)
            stokes_arr[2], stokes_arr[3] = XY.real, XY.imag

        print('Rebinning in time by %d' % width_max)
        mask = list(np.where(stokes_vec[0]==0)[0])
#        mask += range(0,700)
        mask = list(set(mask))
        results_faraday = pol.faraday_fit(stokes_vec, RMmin=inputs.rmmin, 
                                          RMmax=inputs.rmmax, nrm=1000, nphi=200, 
                                          mask=mask, plot=inputs.mk_plot)
        RMs, P_derot_arr, RMmax, phimax, derot_phase = results_faraday

        if inputs.mk_plot:
            plot_RMspectrum(RMs, P_derot_arr, RMmax, 
                            phimax, derot_phase, 
                            fn_fig='%s_RMspectrum.pdf' % obs_name)

        print('Maximum likelihood RM: %0.2f' % RMmax)

        Pcal = (stokes_arr[1]+1j*stokes_arr[2])*derot_phase[:, None]
        stokes_arr[1], stokes_arr[2] = Pcal.real, Pcal.imag

        if inputs.mk_plot:
            plot_all(stoke_arr, suptitle='Faraday derotated')







