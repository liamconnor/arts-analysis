import os

import numpy as np
import matplotlib.pylab as plt
import glob 
import matplotlib.pylab as plt

from darc.sb_generator import SBGenerator
import tools

stokes_ps = ['I', 'Q', 'U', 'V']

trans=False
NFREQ = 1536
freq = (1219.70092773,1519.50561523)
freq_arr = np.linspace(freq[0], freq[-1], NFREQ)
lam_arr = 3e2 / freq_arr
dt = 8.192e-5
rebin_time=1
rebin_freq=1

def make_iquv_arr(dpath, rebin_time=1, rebin_freq=1, 
                  DM=0.0, trans=True, RFI_clean=None):
    """ Read in all 4 arrays, dedisperse, 
    return list with median-subtracted, rebinned 
    [arr_I, arr_Q, arr_U, arr_V]

    Parameters
    ----------
    dpath : str 
        data path with .npy IQUV files
    """
    if type(dpath)==str:
        flist = glob.glob(dpath)
    else:
        flist = dpath

    flist.sort() # should be IQUV ordered now
    arr_list = []

    if type(RFI_clean)==str:
        mask = (np.loadtxt(RFI_clean)).astype(int)
        RFI_clean = True
    else:
        mask = []

    for ii, fn in enumerate(flist):
        print("Assuming %s is Stokes %s" % (fn, stokes_ps[ii]))
        arr = np.load(fn)
        last_ind = -int(abs(4.148e3*DM*(freq[0]**-2-freq[-1]**-2)/dt))

        if trans:
            arr = arr.transpose()
        if arr.shape[0]!=1536:
            print("Are you sure you wanted to transpose?")

        if RFI_clean:
            arr = tools.cleandata(arr, clean_type='perchannel')
            arr[mask] = 0.0

        arr = tools.dedisperse(arr, DM, freq=freq)[:, :last_ind]
        nt, nf = arr.shape[-1], arr.shape[0]
#        arr = arr - np.median(arr, axis=-1, keepdims=True)
#        arr = arr[:nf//rebin_freq*rebin_freq, :nt//rebin_time*rebin_time]
#        arr = arr.reshape(nf//rebin_freq, rebin_freq, 
#                          nt//rebin_time, rebin_time).mean(1).mean(-1)
        arr_list.append(arr)

    pulse_sample = np.argmax(arr_list[0].mean(0))

    return arr_list, pulse_sample

def get_bandpass(stokes_I, alpha=0.51, freq=(1219.70092773,1519.50561523)):
    """ Get bandpass from calibration observation. 
    alpha is powerlaw index, default is 0.51 for 3C286
    """
    freq_arr = np.linspace(freq[0], freq[1], NFREQ)
    bandpass = stokes_I*(freq_arr/1370.0)**alpha
    return bandpass

def sb_from_npy(folder, sb=35, off_src=False):
    # get sb map                                                                                         
    sbgen = SBGenerator.from_science_case(4)
    sbmap = sbgen.get_map(sb)
    # read first file to get shape                                                                       
    shape = np.load('{}/stokesI_tab00.npy'.format(folder)).shape
    
    # init full array                                                                                    
    data_full = np.zeros((12, shape[0], shape[1]))

    for stokes in 'IQUV':
        print("Processing stokes {}".format(stokes))
        for tab in set(sbmap):
            print("Loading TAB{:02d}".format(tab))
            if off_src:
                fn = '{}/stokes{}_tab{:02d}.npy'.format(folder, stokes, tab)
            else:
                fn = '{}/stokes{}_tab{:02d}.npy'.format(folder, stokes, tab)
            data = np.load(fn)
            data_full[tab] = data
        if off_src:
            fnout = 'stokes{}_sb_off'.format(stokes, sb)
        else:
            fnout = 'stokes{}_sb_on'.format(stokes, sb)
        # get sb                                                                                         
        sbdata = sbgen.synthesize_beam(data_full, sb)
        np.save(folder+fnout, sbdata[:])

def calibrate_nonswitch(basedir, src='3C286', save_sol=True):
    """ This function should get both bandpass calibration and 
    a polarisation calibration from some point source (usually 3C286)
    """
    alpha_dict = {'3C286' : 0.51}
    fn_spec = basedir+'/polcal/stokes_uncal_spectra.npy'

    if os.path.exists(fn_spec):
        stokes_arr_spec = np.load(fn_spec)
    else:
        stokes_arr_spec = np.zeros([4, NFREQ])
        for ii, ss in enumerate(stokes_ps):
            don = np.load(basedir+'/polcal/%s/on/stokes%s_sb_on.npy' % (src, ss))
            try:
                doff = np.load(basedir+'/polcal/%s/off/stokes%s_sb_off.npy' % (src, ss))
            except:
                print("There is no polcal off npy file")
                doff = 0*don
        # arr_list, pulse_sample = make_iquv_arr(dpath, rebin_time=1, 
        #                                            rebin_freq=1, dm=0, trans=False,
        #                                            RFI_clean=True)
        # stokes_arr = np.concatenate(arr_list, axis=0)
        # stokes_arr = stokes_arr.reshape(4, NFREQ, -1)
        # stokes_arr_spec = stokes_arr.mean(-1)
            stokes_arr_spec[ii] = don.mean(-1)-doff.mean(-1)

        if save_sol:
            np.save(fn_spec, stokes_arr_spec)

    I = stokes_arr_spec[0]
    Q = stokes_arr_spec[1]
    U = stokes_arr_spec[2]
    V = stokes_arr_spec[3]

    xy = U + 1j*V

    bandpass = get_bandpass(I, alpha=alpha_dict[src])

    if save_sol:
        np.save(basedir+'/polcal/bandpass.npy', bandpass)
        np.save(basedir+'/polcal/xy_phase.npy', np.angle(xy))

    return stokes_arr_spec, bandpass, np.angle(xy)

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
        mask_xy = range(189, 430)+range(821,830)
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
        plt.show()

    return stokes_arr_cal

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
    xy.real -= np.median(xy.real)
    xy.imag -= np.median(xy.imag)
#    if pulse_sample is None:
#        pulse_sample = np.argmax(np.sqrt(arr_U**2 + arr_V**2).mean(0))

#    if pulse_width>1:
#        xy_pulse = xy[:, pulse_sample-pulse_width//2:pulse_sample+pulse_width//2].mean(-1)
#    else:
#        xy_pulse = xy[:, pulse_sample]
    xy_pulse = xy
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

def faraday_fit(stokes_vec, RMmin=-1e4, RMmax=1e4, 
                nrm=5000, nphi=500, mask=None, plot=False):
    """ stokes vec should be (4, NFREQ) array
    """
    phis = np.linspace(0, 2*np.pi, nphi)
    RMs = np.linspace(RMmin, RMmax, nrm)

#    P /= stokes_vec[0].reshape(-1, 4).mean(-1).repeat(4)
    if mask is None:
        use_ind = range(NFREQ)
    else:
        use_ind = np.delete(range(NFREQ), mask)

    Ifit = np.poly1d(np.polyfit(freq_arr[use_ind],
                                stokes_vec[0][use_ind], 10))(freq_arr)

    P = stokes_vec[1] + 1j*stokes_vec[2]
    P.real[use_ind] -= np.median(P.real[use_ind])
    P.imag[use_ind] -= np.median(P.imag[use_ind])
    P /= Ifit

    P_derot_arr = []
    P_derot_arr = np.empty([nrm, nphi])

    if plot:
        fig = plt.figure()
        plt.subplot(411)
        plt.plot(freq_arr[use_ind], stokes_vec[0][use_ind],'.')
        plt.plot(freq_arr[mask], stokes_vec[0][mask],'.')        
        plt.plot(freq_arr, Ifit, color='k')
        plt.ylim(Ifit[use_ind].max()*1.1, Ifit[use_ind].min()*0.9)
        plt.subplot(412)
        plt.plot(freq_arr[use_ind], P.real[use_ind])
        plt.plot(freq_arr[mask], P.real[mask],'.')
        plt.subplot(413)
        plt.plot(freq_arr[use_ind], P.imag[use_ind])
        plt.plot(freq_arr[mask], P.imag[mask],'.')
        plt.subplot(414)
        plt.plot(freq_arr[use_ind], np.angle(P[use_ind]), color='k')
        plt.plot(freq_arr[mask], np.angle(P[mask]), '.', color='C1')
        plt.show()

    for ii, rm in enumerate(RMs):
        for jj, phi in enumerate(phis):
            phase_ = np.exp(-2j*(rm*lam_arr[use_ind]**2)+phi*1j)
            P_derot_arr[ii,jj] = np.sum(P[use_ind]*phase_)

    iimax, jjmax = np.where(P_derot_arr==P_derot_arr.max())
    RMbf = RMs[iimax]
    phibf = phis[jjmax]

    derot_phase = np.exp(-2j*(RMbf*lam_arr**2)+phibf*1j)

    return RMs, P_derot_arr, RMbf, phibf, derot_phase

def plot_raw_data(arr, pulse_sample=None, pulse_width=1):
    """ Plot data before calibration
    """
    fig = plt.figure(figsize=(7,7))

    if pulse_sample is None:
        pulse_sample = np.argmax(arr[0].mean(0))
    
    if len(arr.shape)==3:
        if pulse_width==1:
            arr_spectra = arr[:, :, pulse_sample]
        else:
            arr_spectra = arr[:, :, pulse_sample-pulse_width//2:pulse_sample+pulse_width//2].mean(-1)

    freq_arr = np.linspace(freq[0], freq[-1], arr[0].shape[0])
    plt.subplot(211)
#    [plt.plot(freq_arr, arr_spectra[jj], alpha=0.7, lw=3) for jj in range(4)]
    [plt.plot(arr_spectra[jj], alpha=0.7, lw=3) for jj in range(4)]
    plt.legend(['uncal I','uncal Q','uncal U','uncal V'])

    plt.subplot(212)
    plt.plot(freq_arr, (np.sqrt(arr_spectra[3]**2 + arr_spectra[1]**2 \
                    + arr_spectra[2]**2)/arr_spectra[0]), 
                    color='k', alpha=0.9)
    plt.ylim(0,2)
    plt.xlabel('Freq', fontsize=18)
    plt.ylabel('Pol fraction [uncal]', fontsize=18)

    plt.show()

def plot_im_raw(arr, pulse_sample=None, pulse_width=1):
    if pulse_sample is None:
        pulse_sample = np.argmax(arr[0].mean(0))

    ntime = arr.shape[-1]
    arr = arr[..., :ntime//pulse_width*pulse_width]
    arr = arr.reshape(4, arr.shape[1], -1, pulse_width).mean(-1)

    I = arr[0][:, pulse_sample-50:pulse_sample+50]
    Q = arr[1][:, pulse_sample-50:pulse_sample+50]
    U = arr[2][:, pulse_sample-50:pulse_sample+50]
    V = arr[3][:, pulse_sample-50:pulse_sample+50]

    fig = plt.figure(figsize=(7,7))
    plt.subplot(221)
    plt.ylabel('Freq')    
    plt.imshow(I-np.median(I,axis=-1,keepdims=True), aspect='auto')
    plt.subplot(222)
    plt.imshow(Q-np.median(Q,axis=-1,keepdims=True), aspect='auto')
    plt.subplot(223)
    plt.imshow(U-np.median(U,axis=-1,keepdims=True), aspect='auto')
    plt.xlabel('Time')
    plt.ylabel('Freq')    
    plt.subplot(224)
    plt.imshow(V-np.median(V,axis=-1,keepdims=True), aspect='auto')
    plt.xlabel('Time')
    plt.show()

def solve_muller(Strue, Sobs):
    """ Solve for Mueller matrix using 
    moore-penrose pseudo inverse"""

    M = np.matmul(Sobs, np.linalg.pinv(Strue))
    return M
    

# if __name__=='__main__':
# #        arr, pulse_sample = make_iquv_arr(dpath, rebin_time=1, rebin_freq=1, dm=DM, trans=True)
#     arr = np.load('dedispersed_data.npy')
#     pulse_sample = np.argmax(arr[0].mean(0))

#     bp = np.load('./bandpass_from_3c286_alpha-0.54.npy')
#     xy_phase = np.load('xy_phase_3c286_frequency.npy')
#     xy_phase[200:400] = 2.6
#     xy_cal = np.poly1d(np.polyfit(freq_arr, xy_phase, 7))(freq_arr)

#     mm=200
#     arr = arr[..., pulse_sample-mm:pulse_sample+mm]
#     for ii in range(4):
# #                arr[ii] = arr[ii][:, pulse_sample-mm:pulse_sample+mm]
#             arr[ii] = arr[ii] - arr[ii][:, :150].mean(-1)[:, None]
            
#     I = arr[0]/bp[:,None]
#     Q = arr[1]/bp[:,None]
#     U = arr[2]/bp[:,None]
#     V = arr[3]/bp[:,None]

#     I_rebin = I[:, mm-2:mm+2].mean(-1).reshape(-1, 16).mean(-1).repeat(16)
#     I /= np.median(I[:, mm-2:mm+2].mean(-1))

#     xy_data = (U+1j*V)/I_rebin[:, None]
#     xy_data_cal = xy_data * np.exp(-1j*xy_cal)[:, None]
#     Ucal, Vcal = xy_data_cal.real, xy_data_cal.imag
#     Q /= I_rebin[:, None]

#     #arr, pulse_sample = make_iquv_arr(dpath, rebin_time=1, rebin_freq=1, dm=DM, trans=True)
#     fig = plt.figure(figsize=(7,9))
#     grid = plt.GridSpec(7,3,hspace=0.0,wspace=0.0)

#     tt_arr = np.linspace(0, 2*mm*dt*1e3, 2*mm)
#     plt.subplot(grid[:3, :3])

#     P = Q + 1j*Ucal
#     P *= np.exp(-2j*RM_guess*lam_arr**2)[:, None]

#     plt.plot(tt_arr, I.mean(0)/I.mean(0).max(), color='k', lw=2., alpha=0.7)        
#     plt.plot(tt_arr, np.abs(P.mean(0)), '--', color='red', lw=2., alpha=0.75)
#     plt.plot(tt_arr, np.abs(Vcal).mean(0)-np.abs(Vcal).mean(), '.', color='mediumseagreen', lw=2.)
#     plt.xlim(tt_arr[150], tt_arr[250])
# #        plt.plot(tt_arr, P.mean(0).imag)
#     plt.ylabel('Intensity',labelpad=15, fontsize=13)
#     plt.legend(['I','L','V'], fontsize=11, loc=1)

#     plt.subplot(grid[4:, :3])
#     plt.plot(freq_arr[1::2], I[:, mm-2:mm+2].mean(-1).reshape(-1, 2).mean(-1), color='k',lw=2, alpha=0.7)
#     plt.plot(freq_arr[1::2], Q[:, mm-2:mm+2].mean(-1).reshape(-1, 2).mean(-1), color='C0', alpha=0.65,lw=2)
#     plt.plot(freq_arr[1::2], Ucal[:, mm-2:mm+2].mean(-1).reshape(-1, 2).mean(-1), color='C1',lw=2, alpha=0.6)
#     plt.xlabel('Frequency (MHz)', fontsize=13)
#     plt.ylabel('Intensity',labelpad=15, fontsize=13)
#     plt.ylim(-1., 3.1)
#     plt.xlim(1205.0, 1550)
#     plt.legend(['I','Q','U'], framealpha=1.0, fontsize=11, loc=(0.88, 0.6))
#     plt.grid('on', alpha=0.25)

#     plt.subplot(grid[3, :3])
#     plt.plot(tt_arr, 180./np.pi*np.angle(P.mean(0)),'.', color='k')
#     plt.xlim(tt_arr[150], tt_arr[250])
#     plt.ylabel('PA (deg)', fontsize=13)
#     plt.xlabel('Time (ms)', labelpad=1, fontsize=13)

#     plt.tight_layout()
#     plt.savefig('./FRB191108_polarisation.pdf')
#     plt.show()
#     exit()
# #        plot_raw_data(arr)
# #        plot_im_raw(arr)
#     Ucal, Vcal, xy_cal, phi_bf = derotate_UV(arr[2], arr[3], pulse_sample=pulse_sample, pulse_width=1)
#     print(Ucal.shape, arr[1].shape, xy_cal.shape)
#     Ucal = xy_cal.real
#     Qcal, Ucal, P_cal, rm_bf, lam_arr, phase_std, P = derotate_faraday(arr[1], Ucal, pulse_sample=pulse_sample, 
#         pulse_width=1, RMmin=0.0, RMmax=1e4)
