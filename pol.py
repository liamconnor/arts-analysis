import os

import numpy as np
import matplotlib.pylab as plt
import glob
import matplotlib.pylab as plt

try:
    from darc.sb_generator import SBGenerator
except:
    print("Could not load DARC")
    SBGenerator = None

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
                  DM=0.0, trans=True, RFI_clean=None, mjd=None):
    """ Read in all 4 arrays, dedisperse,
    return list with median-subtracted, rebinned
    [arr_I, arr_Q, arr_U, arr_V]

    Parameters
    ----------
    dpath : str
        data path with .npy IQUV files
    """
    if type(dpath)==str:
        if mjd is not None:
            flist = glob.glob(dpath + '/*mjd{}*'.format(mjd))
        else:
            flist = glob.glob(dpath)
    else:
        flist = dpath

    flist.sort() # should be IQUV ordered now
    arr_list = []
    print("IQUV arrays:", flist)

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
            print("CLEANING RFI")
            arr = tools.cleandata(arr, clean_type='perchannel', n_iter_time=3)
            arr[mask] = 0.0

        arr = tools.dedisperse(arr, DM, freq=freq, K=1/2.410e-4)[:, :last_ind]
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

def sb_from_npy(folder, sb=35, off_src=False, mjd=None):
    # get sb map
    sbgen = SBGenerator.from_science_case(4)
    sbmap = sbgen.get_map(sb)
    # read first file to get shape
    if mjd is not None:
        print(mjd)
        fn_ = '%s/stokesI_tab00_mjd%0.7f.npy' % (folder, mjd)
        shape = np.load(fn_).shape
        if os.path.exists(fn_):
            shape = np.load(fn_).shape
        else:
            print("{} does not exist".format(fn_))
            exit()
    else:
        fn_ = '{}/stokesI_tab00.npy'.format(folder)
        if os.path.exists(fn_):
            shape = np.load(fn_).shape
        else:
            print("{} does not exist".format(fn_))
            exit()

    # init full array
    data_full = np.zeros((12, shape[0], shape[1]))

    for stokes in 'IQUV':
        print("Processing stokes {}".format(stokes))
        for tab in set(sbmap):
            print("Loading TAB{:02d}".format(tab))
            if off_src:
                if mjd is not None:
                    fn = '{}/stokes{}_tab{:02d}_mjd{:.7f}.npy'.format(folder, stokes, tab, mjd)
                else:
                    fn = '{}/stokes{}_tab{:02d}.npy'.format(folder, stokes, tab)
            else:
                if mjd is not None:
                    fn = '{}/stokes{}_tab{:02d}_mjd{:.7f}.npy'.format(folder, stokes, tab, mjd)
                else:
                    fn = '{}/stokes{}_tab{:02d}.npy'.format(folder, stokes, tab)
            data = np.load(fn)
            data_full[tab] = data
        if off_src:
            if mjd is not None:
                fnout = 'stokes{}_mjd{:.6f}_sb{}_off'.format(stokes, mjd, sb)
            else:
                fnout = 'stokes{}_sb{}_off'.format(stokes, sb)
        else:
            if mjd is not None:
                fnout = 'stokes{}_mjd{:.6f}_sb{}_on'.format(stokes, mjd, sb)
            else:
                fnout = 'stokes{}_sb{}_on'.format(stokes, sb)
        # get sb
        sbdata = sbgen.synthesize_beam(data_full, sb)
        np.save(folder+fnout, sbdata[:])

def calibrate_GxGy(folder_polcal, src='3C147', save_sol=True):
    fn_spec = folder_polcal+'/stokes_uncal_spectra{}.npy'.format(src)

    if os.path.exists(fn_spec):
        stokes_arr_spec = np.load(fn_spec)
    else:
        stokes_arr_spec = np.zeros([4, NFREQ])
        for ii, ss in enumerate(stokes_ps):
            print(basedir+'/{0}/on/stokes{1}_sb{2}_on.npy'.format(src, ss, sb))
            don = np.load(basedir+'/{0}/on/stokes{1}_sb{2}_on.npy'.format(src,
                    ss, sb))
            try:
                doff = np.load(basedir
                        +'/{0}/off/stokes{1}_sb{2}_off.npy'.format(src,
                        ss, sb))
            except:
                print("There is no polcal off npy file")
                doff = 0*don

            stokes_arr_spec[ii] = (don.mean(-1)-doff.mean(-1))/np.std(doff.mean(0))

        if save_sol:
            np.save(fn_spec, stokes_arr_spec)

    I = stokes_arr_spec[0]
    Q = stokes_arr_spec[1]
    U = stokes_arr_spec[2]
    V = stokes_arr_spec[3]

#   xy = U + 1j*V

    if save_sol:
        np.save(basedir+'/bandpass_{}.npy'.format(src), bandpass)
#        np.save(basedir+'/xy_phase.npy', np.angle(xy))

    return stokes_arr_spec, bandpass, np.angle(xy)

def calibrate_linpol(basedir, src='3C286', sb=35, save_bandpass=True, 
                     save_xyphase=True):
    """ This function should get both bandpass calibration and
        a polarisation calibration from some linearly polarised
        point source (usually 3C286)
    """
    alpha_dict = {'3C286' : 0.51,
                  '3C147' : 0.66}
    fn_spec = basedir+'/stokes_uncal_spectra{}.npy'.format(src)

    if os.path.exists(fn_spec):
        stokes_arr_spec = np.load(fn_spec)
    else:
        stokes_arr_spec = np.zeros([4, NFREQ])
        for ii, ss in enumerate(stokes_ps):
            fn_=basedir+'/{0}/on/stokes{1}_sb{2}_on.npy'.format(src, ss, sb)
            assert os.path.exists(fn_), "Missing {}. Try using -sb?".format(fn_)
            don = np.load(fn_)
            try:
                doff = np.load(basedir
                        +'/{0}/off/stokes{1}_sb{2}_off.npy'.format(src,
                        ss, sb))
            except:
                print("There is no polcal off npy file")
                doff = 0*don
            stokes_arr_spec[ii] = (don.mean(-1)-doff.mean(-1))/np.std(doff.mean(0))

            np.save(fn_spec, stokes_arr_spec)

    I = stokes_arr_spec[0]
    Q = stokes_arr_spec[1]
    U = stokes_arr_spec[2]
    V = stokes_arr_spec[3]

    xy = U + 1j*V

    bandpass = get_bandpass(I, alpha=alpha_dict[src])

    if save_bandpass:
        np.save(basedir+'/bandpass_{}.npy'.format(src), bandpass)
    if save_xyphase:
        np.save(basedir+'/xy_phase_{}.npy'.format(src), np.angle(xy))

    return stokes_arr_spec, bandpass, np.angle(xy)

def xy_gain_ratio(stokes_arr_unpol, fngxgy_out=None):
    """ Estimate the ratio of X pol gain to Y pol gain
    using an unpolarised point source like 3C147.
    """
    if len(stokes_arr_unpol.shape)==3:
        stokes_arr_unpol = stokes_arr_unpol.mean(-1)
    elif len(stokes_arr_unpol.shape)==1:
        print("Stokes array should be length 2 or 3 (4, nfreq, ntime)")
        exit()

    I = stokes_arr_unpol[0]
    Q = stokes_arr_unpol[1]
    fgxgy = (I-Q)/(I+Q)
    # replace zeros, nans, and np.inf with 1
    fgxgy[fgxgy==0] = 1.
    fgxgy[fgxgy==np.inf] = 1.
    fgxgy[fgxgy==np.nan] = 1.

    if fngxgy_out is not None:
        np.save(fngxgy_out, fgxgy)
    else:
        print("Gain array cannot be saved, filename not defined")

    return fgxgy

def calibrate(stokes_arr, xyphase, fn_GxGy):

    if type(xyphase)==str:
        xyphase = np.load(xyphase)[..., None]
    elif len(xyphase.shape)==1:
        xyphase = xyphase[:, None]

    if len(stokes_arr.shape)==2:
        stokes_arr = stokes_arr[..., None]

    stokes_arr_out = np.zeros_like(stokes_arr)

    I, Q, U, V = stokes_arr[0], stokes_arr[1], stokes_arr[2], stokes_arr[3]
    xy_data = U + 1j*V
    xy_data *= np.exp(-1j*xyphase)
    
    stokes_arr_cal_IQ = unleak_IQ(stokes_arr, fn_GxGy)[0]
    stokes_arr_out[0] = stokes_arr_cal_IQ[0]
    stokes_arr_out[1] = stokes_arr_cal_IQ[1]
    stokes_arr_out[2] = xy_data.real
    stokes_arr_out[3] = xy_data.imag
    
    return stokes_arr_out

def unleak_IQ(stokes_arr, fn_GxGy):
    """ Take array of gain ratios (fn_GxGy, either
    path or numpy array) and remove leakage from
    I to Q.
    """

    if type(fn_GxGy)==str:
        fGxGy = np.load(fn_GxGy)
    else:
        fGxGy = fn_GxGy

    if len(stokes_arr.shape)==2:
        stokes_arr = stokes_arr[..., None]

    stokes_arr_cal = np.zeros_like(stokes_arr)

    nfreq, ntime = stokes_arr[0].shape
    assert nfreq==len(fGxGy), "Cal array different length from data"

    # Construct coherence matrixes
    Pobs = np.zeros([2,2,nfreq,ntime])
    Ptrue = np.zeros([2,2,nfreq,ntime])
    Pobs[0,0] = stokes_arr[0]+stokes_arr[1] # I+Q
#    Pobs[:,0,1] = stokes_arr[2]+1j*stokes_arr[3] # U+iV
#    Pobs[:,1,0] = stokes_arr[3]+stokes_arr[1] # U-iV
    Pobs[1,1] = stokes_arr[0]-stokes_arr[1] # I-Q

    for ii in xrange(nfreq):
        # Construct rotation matrix
        M = np.array([[1,0],[0,fGxGy[ii]]])
        Minv = np.linalg.inv(M)
        # Derotate Q/I
        Ptrue[:,:,ii] = np.dot(Minv, Pobs[:,:,ii])

    stokes_arr_cal[0] = 0.5 * (Ptrue[0,0]+Ptrue[1,1])
    stokes_arr_cal[1] = 0.5 * (Ptrue[0,0]-Ptrue[1,1])
    stokes_arr_cal[2] = stokes_arr[2]
    stokes_arr_cal[3] = stokes_arr[3]

#    for ii in range(4):
#        stokes_arr_cal[ii] -= np.median(stokes_arr_cal[ii])
#        stokes_arr_cal[ii] /= np.std(np.mean(stokes_arr_cal[ii],axis=0))

    stokes_arr_cal[np.isnan(stokes_arr_cal)] = 0.0

    return stokes_arr_cal, fGxGy


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
        mask_xy = range(11,13)+range(88,94)+range(39,43)+range(189,430)+\
                  range(821,830)+range(1530,1536)
        use_ind_xy = np.delete(use_ind_xy, mask_xy)

#    xy_cal = np.poly1d(np.polyfit(freq_arr[use_ind_xy],
#                    xy_phase[use_ind_xy], 14))(freq_arr)

    print("Removing polyfit: test")
    xy_cal = np.zeros([xy_phase.shape[0]])
    xy_cal[use_ind_xy] = xy_phase[use_ind_xy]

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
        plt.plot(mask_xy, xy_phase[mask_xy],'.')
        plt.plot(xy_cal, color='red')
        plt.xlabel('Freq channel')
        plt.ylabel('xy phase (rad)')
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
        plt.legend(["Stokes I","Mask","Fit"])
        plt.subplot(412)
        plt.plot(freq_arr[use_ind], P.real[use_ind])
        plt.plot(freq_arr[mask], P.real[mask],'.')
        plt.legend(["Stokes Q","Mask"])
        plt.subplot(413)
        plt.plot(freq_arr[use_ind], P.imag[use_ind])
        plt.plot(freq_arr[mask], P.imag[mask],'.')
        plt.legend(["Stokes U","Mask"])
        plt.subplot(414)
        plt.plot(freq_arr[use_ind], np.angle(P[use_ind]), color='k')
        plt.plot(freq_arr[mask], np.angle(P[mask]), '.', color='C1')
        plt.legend(["phase(Q/U)","Mask"])
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
    plt.imshow(I-np.median(I,axis=-1,keepdims=True), aspect='auto', cmap='RdBu')
    plt.subplot(222)
    plt.imshow(Q-np.median(Q,axis=-1,keepdims=True), aspect='auto', cmap='RdBu')
    plt.subplot(223)
    plt.imshow(U-np.median(U,axis=-1,keepdims=True), aspect='auto', cmap='RdBu')
    plt.xlabel('Time')
    plt.ylabel('Freq')
    plt.subplot(224)
    plt.imshow(V-np.median(V,axis=-1,keepdims=True), aspect='auto', cmap='RdBu')
    plt.xlabel('Time')
    plt.show()

def solve_muller(Strue, Sobs):
    """ Solve for Mueller matrix using
    moore-penrose pseudo inverse"""

    M = np.matmul(Sobs, np.linalg.pinv(Strue))
    return M
