import os

import numpy as np
import matplotlib.pyplot as plt

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

import argparse 
import glob
import time

import tools
import pol
import read_IQUV_dada

freq_arr = pol.freq_arr
pulse_width = 1 # number of samples to sum over
transpose = False
SNRtools = tools.SNR_Tools()

def generate_iquv_arr(dpath, dedisp_data_path='', DM=0, rfimask=None):
    if dedisp_data_path=='' or not os.path.exists(dedisp_data_path):
        print("No dedispersed file, using %s" % dpath)
        if len(glob.glob(dpath))==0:
            print("You still need to generate the SB files! Try -sb")
            exit()
        arr_list, pulse_sample = pol.make_iquv_arr(dpath, 
                                                   DM=DM, 
                                                   trans=False,
                                                   RFI_clean=rfimask)
        stokes_arr = np.concatenate(arr_list, axis=0)
        stokes_arr = stokes_arr.reshape(4, pol.NFREQ, -1)
        stokes_arr = stokes_arr[..., pulse_sample-5000:pulse_sample+5000]

        if type(dedisp_data_path)!='':
            np.save(dedisp_data_path, stokes_arr)

    elif os.path.exists(dedisp_data_path):
        print("Reading %s in directly" % dedisp_data_path)
        stokes_arr = np.load(dedisp_data_path)
        pulse_sample = np.argmax(stokes_arr[0].mean(0))
        stokes_arr = stokes_arr[..., pulse_sample-500:pulse_sample+500]
        pulse_sample = 500

        if type(rfimask)==str:
            mask = np.loadtxt(rfimask).astype(int)
            stokes_arr[:, mask] = 0.0
    else:
        print("Could not generate IQUV array")
        exit()
    for ii in range(4):
         ind = np.where(stokes_arr[ii]!=0)[0]
         sigstokes = np.std(np.mean(stokes_arr[ii][ind], axis=0))
         stokes_arr[ii] /= sigstokes

    return stokes_arr, pulse_sample

def read_dedisp_data(dpath):
    stokes_arr = np.load(dedisp_data_path)
    pulse_sample = np.argmax(stokes_arr[0].mean(0))

    return stokes_arr, pulse_sample

def get_width(data):
    snr_max, width_max = SNRtools.calc_snr_matchedfilter(data, 
            widths=range(500))
    return snr_max, width_max

def plot_dedisp(stokes_arr, pulse_sample=None, stokes='iquv', 
                pulse_width=1, params=None, outdir=None):
    #stokes_arr = stokes_arr[..., :len(stokes_arr[-1])//pulse_width*pulse_width]
    #stokes_arr = stokes_arr.reshape(4, -1, stokes_arr.shape[-1]//pulse_width, pulse_width).mean(-1)
    stokes = stokes.upper()

    if pulse_sample is None:
        pulse_sample = np.argmax(stokes_arr[0].mean(0))
    
    I = stokes_arr[0]
    Q = stokes_arr[1]
    U = stokes_arr[2]
    V = stokes_arr[3]
    L = np.sqrt(Q**2 + U**2)
    Ptotal = np.sqrt((Q-np.mean(Q))**2 + (U-np.mean(U))**2 + 
             (V-np.mean(V))**2).mean(0)
    Ptotal -= np.median(Ptotal)

    if params:
        suptitle = '{0} {1} {2}'.format(params["SOURCE"], stokes, params["UTC"])

    if stokes == 'IQUV':
        plt.subplot(211)
        plt.plot(I.mean(0)-I.mean())
        plt.plot(Q.mean(0)-Q.mean())
        plt.plot(U.mean(0)-U.mean())
        plt.plot(V.mean(0)-V.mean())
        #plt.plot(Ptotal,'--',color='k')
        plt.legend(['I', 'Q', 'U', 'V'])#, 'Pol total'])
        plt.subplot(212)
        plt.plot(I.mean(0)-I.mean())
        plt.plot(Q.mean(0)-Q.mean())
        plt.plot(U.mean(0)-U.mean())
        plt.plot(V.mean(0)-V.mean())
        #plt.plot(Ptotal,'--',color='k')
        plt.xlim(pulse_sample-50, pulse_sample+50)
        plt.xlabel('Sample number', fontsize=15)
        plt.suptitle(suptitle)

    elif stokes == 'ILV':
        plt.subplot(211)
        plt.plot(I.mean(0)-I.mean())
        plt.plot(L.mean(0)-L.mean())
        plt.plot(V.mean(0)-V.mean())
        plt.legend(['I', 'L', 'V'])#, 'Pol total'])
        plt.subplot(212)
        plt.plot(I.mean(0)-I.mean())
        plt.plot(L.mean(0)-L.mean())
        plt.plot(V.mean(0)-V.mean())
        plt.xlim(pulse_sample-50, pulse_sample+50)
        plt.xlabel('Sample number', fontsize=15)
        plt.suptitle(suptitle)

    if outdir is not None:
        fn_fig = outdir + '/{}_dedispersed_{}_{}.pdf'.format(params["SOURCE"],
                stokes, params["MJD"])
        plt.savefig(fn_fig, pad_inches=0, bbox_inches='tight')
    plt.show()

def create_params_txt(outdir, fndada):
    """Example dictionary: {BEAM: 25.0,
       EVENT_BEAM: 68.0, 
       EVENT_DM: 246.800003,
       EVENT_SNR: 14.8403,
       EVENT_WIDTH: 10.0}"""
#    fname = 'DM588.13_SNR60_CB21_SB37_Width5.txt'
    event_params = read_IQUV_dada.read_event_params(fndada)
    vals = (event_params["EVENT_DM"], 
            event_params["MJD"],
            event_params["EVENT_SNR"], 
            event_params["BEAM"], 
            event_params["EVENT_BEAM"], 
            event_params["EVENT_WIDTH"])
    fname = 'DM%0.2f_MJD%0.6f_SNR%d_CB%d_SB%d_Width%d.txt' % vals
    os.system('touch %s/%s' % (outdir, fname))

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

def plot_RMspectrum(RMs, P_derot_arr, RMmax, phimax, derot_phase, 
                    fn_fig=None, params=None, outdir=None):
    """ Plot power vs. RM 
    """
    fig=plt.figure()
    plt.plot(RMs, np.max(P_derot_arr, axis=-1))
    plt.title('RM spectrum {0} {1}'.format(params["SOURCE"], params["UTC"]))
    plt.ylabel('Defaraday amplitude', fontsize=16)
    plt.xlabel(r'RM (rad m$^{-2}$)', fontsize=16)
    plt.axvline(RMmax, color='r', linestyle='--')
    plt.text(RMmax*0.5, np.max(P_derot_arr)*0.8, '%s \n RMmax~%d' % (obs_name, RMmax))
    if outdir is not None:
        fn_fig = outdir + '/{}_RM_spectrum_{}.pdf'.format(params["SOURCE"], 
                                                          params["MJD"])
        plt.savefig(fn_fig, pad_inches=0, bbox_inches='tight')
    fig=plt.figure()
    extent=[0, 360, inputs.rmmin, inputs.rmmax]
    plt.imshow(P_derot_arr, aspect='auto', vmax=P_derot_arr.max(), 
              vmin=P_derot_arr.max()*0.5, extent=extent, cmap='RdBu')
    plt.xlabel('Phi (deg)', fontsize=16)
    plt.ylabel('RM (rad m**-2)', fontsize=16)
    plt.show()

def rebin_tf(data, tint=1, fint=1):
    """ Rebin in time and frequency accounting 
    for zeros
    """
    if len(data.shape)==3:
        dsl=3
        nfreq, ntime = data[0].shape
    elif len(data.shape)==2:
        dsl=2
        nfreq, ntime = data.shape
        data = data[None]

    if fint>1:
        # Rebin in frequency
        data_ = data[:,:nfreq//fint*fint].reshape(-1, nfreq//fint, fint, ntime)
        weights = (data_.mean(-1)>0).sum(2)
        data_ = np.sum(data_, axis=2) / weights[:,:,None]
        data_[np.isnan(data_)] = 0.0
    else:
        data_ = data

    if tint>1:
        # Rebin in time
        data_ = data_[:, :, :ntime//tint*tint].reshape(-1, 
                nfreq//fint, ntime//tint, tint)
        weights = (data_.mean(1)>0).sum(-1)
        data_ = np.sum(data_, axis=-1) / weights[:, None]
        data_[np.isnan(data_)] = 0.0

    return data_

def plot_all(stokes_arr, suptitle='', fds=16, tds=1, stokes='IQUV', 
        params=None, outdir=None):
    """ Create a 4x2 grid of subplots for 
    pulse profiles and waterfalls in all 
    four Stokes params
    """
    stokes = stokes.upper()
    time_mid = int(stokes_arr.shape[-1]//2)
#    stokes_arr = rebin_tf(stokes_arr, tint=fds, fint=tds)
    stokes_arr_ = stokes_arr[:,:,time_mid-200:time_mid+200]
    stokes_arr_ = stokes_arr_.reshape(4,1536//fds,fds,-1).mean(2)
    stokes_arr_ = stokes_arr_[...,:stokes_arr_.shape[-1]//tds*tds]
    stokes_arr_ = stokes_arr_.reshape(4,1536//fds,-1,tds).mean(-1)
    print(stokes_arr_.shape)
    for ii in range(4):
        stokes_arr_[ii] -= np.median(stokes_arr_[ii], axis=-1, keepdims=True)
        stokes_arr_[ii] /= np.std(stokes_arr_[ii], axis=-1, keepdims=True)

    stokes_arr_[np.isnan(stokes_arr_)] = 0.0

    if params:    
        title = suptitle + ' {0} {1} {2}'.format(stokes, params["SOURCE"], 
                params["UTC"])
    
    if stokes == 'IQUV':
        plt.subplot(421)
        plt.plot(stokes_arr_[0].mean(0))
        plt.ylabel('I')
        plt.subplot(422)
        plt.imshow(stokes_arr_[0], aspect='auto',vmax=3,vmin=-2, cmap='RdBu')
        plt.ylabel('Freq')
        plt.subplot(423)
        plt.plot(stokes_arr_[1].mean(0))
        plt.ylabel('Q')
        plt.subplot(424)
        plt.imshow(stokes_arr_[1], aspect='auto',vmax=3,vmin=-2, cmap='RdBu')
        plt.ylabel('Freq')
        plt.subplot(425)
        plt.plot(stokes_arr_[2].mean(0))
        plt.ylabel('U')
        plt.subplot(426)
        plt.imshow(stokes_arr_[2], aspect='auto',vmax=3,vmin=-2, cmap='RdBu')
        plt.ylabel('Freq')
        plt.subplot(427)
        plt.plot(stokes_arr_[3].mean(0))
        plt.ylabel('V')
        plt.subplot(428)
        plt.imshow(stokes_arr_[3], aspect='auto',vmax=3,vmin=-2, cmap='RdBu')
        plt.ylabel('Freq')
        plt.xlabel('Time (samples)')
        plt.suptitle(title)

    elif stokes == 'ILV':
        stokes_arr_l = (np.sqrt(stokes_arr_[1]**2 + stokes_arr_[2]**2))
        plt.subplot(321)
        plt.plot(stokes_arr_[0].mean(0))
        plt.ylabel('I')
        plt.subplot(322)
        plt.imshow(stokes_arr_[0], aspect='auto',vmax=3,vmin=-2, cmap='RdBu')
        plt.ylabel('Freq')
        plt.subplot(323)
        plt.plot(stokes_arr_l.mean(0))
        plt.ylabel('L')
        plt.subplot(324)
        plt.imshow(stokes_arr_l, aspect='auto',vmax=3,vmin=-2, cmap='RdBu')
        plt.ylabel('Freq')
        plt.subplot(325)
        plt.plot(stokes_arr_[3].mean(0))
        plt.ylabel('V')
        plt.subplot(326)
        plt.imshow(stokes_arr_[3], aspect='auto',vmax=3,vmin=-2, cmap='RdBu')
        plt.ylabel('Freq')
        plt.xlabel('Time (samples)')
        plt.suptitle(title)  

    if outdir is not None:
        fn_fig = outdir + '/{}_{}_stokes{}_{}.pdf'.format(params["SOURCE"], 
                suptitle, stokes, params["MJD"])
        plt.savefig(fn_fig, pad_inches=0, bbox_inches='tight')
    plt.show()


def mk_pol_plot(stokes_arr, pulse_sample=None, pulse_width=1, params=None):
    if pulse_sample is None:
        pulse_sample = np.argmax(stokes_arr[0].mean(0))
    pol.plot_im_raw(stokes_arr, pulse_sample=pulse_sample, 
            pulse_width=pulse_width)
    pol.plot_raw_data(stokes_arr, pulse_sample=pulse_sample, 
            pulse_width=pulse_width)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs polarisation pipeline",
                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--basedir', 
                        help='base directory of polarisation data', 
                        type=str, required=True)
    parser.add_argument('-A', '--All', 
                        help='Do every step in the pipeline', 
                        action='store_true')
    parser.add_argument('-dd', '--dada', 
                        help='generate numpy files from dada',
                        action='store_true')
    parser.add_argument('-sb', '--gen_sb', 
                        help='generate SB from npy files', 
                        action='store_true')
    parser.add_argument('-g', '--gen_arr', 
                        help='generate iquv array', action='store_true')
    parser.add_argument('-pc', '--polcal', 
                        help='do polarisation calibration', 
                        action='store_true')
    parser.add_argument('-c', '--calibrate_frb', 
                        help='use non-switch polcal solution to cal FRB', 
                        action='store_true')
    parser.add_argument('-F', '--faraday', 
                        help='Faraday fit and de-rotate', 
                        action='store_true')
    parser.add_argument('-p', '--mk_plot', 
                        help='plot various data products along the way', 
                        action='store_true')
    parser.add_argument('-s', '--stokes',
                        help='stokes to plot if mk_plot==True. (iquv|ilv)',
                        default='iquv', type=str)
    parser.add_argument('-pcd', '--polcal_dir',
                        help='path to polcal files', default=None, type=str)
    parser.add_argument('-src_linpol', '--src_linpol', help='calibrator source name', 
                        default='3C286', type=str)
    parser.add_argument('-src_unpol', '--src_unpol', help='calibrator source name', 
                        default='3C147', type=str)
    parser.add_argument('-rmmin', '--rmmin', help='min RM to search', 
                        default=-1e4, type=float)
    parser.add_argument('-rmmax', '--rmmax', help='max RM to search', 
                        default=1e4, type=float)
    parser.add_argument('-fds', '--freq_downsample', help='downsample in freq',
                        default=16, type=int)
    parser.add_argument('-tds', '--time_downsample', 
                        help='downsample in freq', default=1, type=int)
    parser.add_argument('-b', '--burst_number', help='burst number',
                        default=0, type=int)

    
    inputs = parser.parse_args()
    #obs_name = inputs.basedir.split('/')[4]

    check_dirs(inputs.basedir)

    if inputs.polcal or inputs.All:
        # Defining polcal directory
        if inputs.polcal_dir is None:
            folder_polcal = inputs.basedir+'/polcal/'
        else:
            folder_polcal = inputs.polcal_dir

        if os.path.isdir(os.path.join(folder_polcal, inputs.src_linpol)):
            src_linpol = inputs.src_linpol
        else:
            print("WARNING: No linearly polarised source")

        if os.path.isdir(os.path.join(folder_polcal, inputs.src_unpol)):
            src_unpol = inputs.src_unpol
        else:
            print("WARNING: No unpolarised source")

        # Defining calibration sources from polcal folder
        # cal_srcs = [src for src in os.listdir(folder_polcal) 
        #         if os.path.isdir(os.path.join(folder_polcal, src))]

        # if '3C286' in cal_srcs:
        #     src_pol = '3C286'
        # else:
        #     print("WARNING: No polarised source")
        #     src_pol = None

        # if '3C147' in cal_srcs:
        #     src_unpol = '3C147'
        # else:
        #     print("WARNING: No unpolarised source")
        #     src_unpol = None

    if inputs.dada or inputs.All:
        print("ARE YOU SURE YOU WANT TO USE ALL THAT MEMORY for dada writer?")
        print("Sleeping for 5 seconds to let you decide.")
        time.sleep(5)
        fndada_all = glob.glob(inputs.basedir+'/dada/*dada')
        fndada_all.sort()
        fndada = fndada_all[inputs.burst_number]
        print('Reading dada file ' + fndada)
        outdir = inputs.basedir+'/numpyarr/'
        event_params = read_IQUV_dada.read_event_params(fndada)
        create_params_txt(outdir, fndada)

        #if len(glob.glob(outdir+'/*npy'))<6:
        if len(glob.glob(outdir+'/*{}*npy'.format(event_params['MJD'])))<10:
            print("Converting dada into numpy for %s" % fndada)
            os.system('echo ./read_IQUV_dada.py %s --outdir %s --mjd %s' 
                      % (fndada, outdir, event_params['MJD']))
        else:
            print("No need to generate numpy from dada for mjd:{} \
                    because they already exist".format(event_params['MJD']))

        if inputs.polcal:
          for src in [src_linpol, src_unpol]:
            if src is not None:
                print(src) 
                print("\n\nGenerating {} on source npy from dada".format(src))
                fndada = glob.glob(folder_polcal + '/%s/on/*dada' % src)[0]
                outdir = folder_polcal+'/%s/on/' % src
                os.system('./read_IQUV_dada.py %s --outdir %s' % (fndada,outdir))

                print("\n\nGenerating {} off source npy from dada".format(src))
                fndada = glob.glob(folder_polcal+'/%s/off/*dada' % src)[0]
                outdir = folder_polcal+'/%s/off/' % src
                os.system('./read_IQUV_dada.py %s --outdir %s' % (fndada,outdir))


    else:
        fndada_all = glob.glob(inputs.basedir+'/dada/*dada')
        fndada_all.sort()
        fndada = fndada_all[inputs.burst_number]
        event_params = read_IQUV_dada.read_event_params(fndada)

    try:
        params = glob.glob(inputs.basedir+'/numpyarr/DM*txt')[0]
    except:
        print("Expected a txt file with DM, width, CB, and SB in %s" % 
            (inputs.basedir+'/numpyarr/'))
        print("e.g. numpyarr/DM588.13_MJDXXX.XXX_SNR60_CB21_SB37_Width5.txt")
        exit()

    DM = event_params["EVENT_DM"]
    SB = int(event_params["EVENT_BEAM"])
    MJD = event_params["MJD"]
    obs_name = event_params["SOURCE"]

    print("\tSOURCE = {}".format(obs_name))
    print("\tDM     = {}".format(DM))
    print("\tSB     = {}".format(SB))
    print("\tMJD    = {}".format(MJD))

    if inputs.gen_sb or inputs.All:
        print("Generating SB=%d from npy data" % SB)
        if inputs.polcal or inputs.All:
          for src in [src_linpol, src_unpol]:
            if src is not None:
              print("Generating SB for %s" % src)
              pol.sb_from_npy(folder_polcal+'/%s/on/' % src, sb=35, 
                      off_src=False)
              pol.sb_from_npy(folder_polcal+'/%s/off/' % src, sb=35, 
                      off_src=True)

        folder = inputs.basedir+'/numpyarr/'
    
        #if len(glob.glob(folder+'stokes*_on*'))<4:
        if len(glob.glob(folder+'stokes*{}*_on*'.format(MJD)))<4:
            pol.sb_from_npy(folder, sb=SB, off_src=False, mjd=MJD)
        else:
            print("Wait, no, SB files are already there!")

    if inputs.polcal or inputs.All:
        #if inputs.polcal or inputs.All:
        #    if inputs.polcal_dir is None:
        #        folder_polcal = inputs.basedir+'/polcal/'
        #    else:
        #        folder_polcal = inputs.polcal_dir
        print("Polcal folder %s" % folder_polcal)
        print("Getting bandpass and xy pol solution from %s" % src_linpol)
        stokes_arr_spec, bandpass, xy_phase = pol.calibrate_nonswitch(
                folder_polcal, src=src_linpol, save_sol=True)

    if inputs.gen_arr or inputs.All:
        fnmask = inputs.basedir+'/numpyarr/rfimask'

        if not os.path.exists(fnmask):
            # There is no rfimask file, but will do rfi cleaning
            rfimask = True
        else:
            # Mask out certain channels
            rfimask = fnmask

        print("Assuming %0.2f for %s" % (DM, obs_name))
        dpath = (inputs.basedir 
                + '/numpyarr/stokes*mjd{:.6f}*sb*.npy'.format(MJD))
        flist_sb = glob.glob(dpath)
        if len(flist_sb) == 0:
            dpath = inputs.basedir + '/numpyarr/stokes*sb*.npy'
            flist_sb = glob.glob(dpath)
        dedisp_data_path = (inputs.basedir 
                + '/numpyarr/{}_mjd{:.6f}_dedisp.npy'.format(obs_name, MJD))

        if len(flist_sb)==-1:
            pass
        else:
            if not os.path.exists(dedisp_data_path):
                 fn_dedisp = inputs.basedir + \
                         '/numpyarr/*_mjd{:.6f}_dedisp.npy'.format(MJD)
                 try:
                     dedisp_data_path = glob.glob(fn_dedisp)[0]
                 except:
                     dedisp_data_path = inputs.basedir +\
                             '/numpyarr/{}_mjd{:.6f}_dedisp.npy'.format(
                             obs_name, MJD)

            stokes_arr, pulse_sample = generate_iquv_arr(dpath, 
                    dedisp_data_path=dedisp_data_path, DM=DM, rfimask=rfimask)
            snr_max, width_max = get_width(stokes_arr[0].mean(0))

    try:
        stokes_arr
    except:
        print('stokes_arr not defined')
        exit()

    if inputs.mk_plot or inputs.All:
        plotdir = inputs.basedir+'/plots'
        print("Plots saved in " + plotdir)
        try:
           os.mkdir(plotdir)
        except OSError:
           print("Directory", plotdir, "exists")
        try:
           stokes_arr
        except NameError:
           print("Cannot plot data if there is no stokes array")
           exit()
        plot_dedisp(stokes_arr, pulse_sample=pulse_sample, 
                    pulse_width=width_max, stokes=inputs.stokes, 
                    params=event_params, outdir=plotdir)

    if inputs.polcal or inputs.All:
        try:
           stokes_arr
        except NameError:
           print("Cannot calibrate FRB if there is no stokes array")
           exit()

        fn_bandpass = folder_polcal+'/bandpass.npy'
        fn_xy_phase = folder_polcal+'/xy_phase.npy'
        fn_GxGy = folder_polcal+'/GxGy_ratio_freq.npy'

        if os.path.exists(fn_GxGy):
            print("\n\nRemoving XY gain difference")
            stokes_arr_cal, fGxGy = pol.unleak_IQ(stokes_arr.copy(), fn_GxGy)
        else:
            stokes_arr_cal = stokes_arr.copy()

        print("Calibrating bandpass")
        stokes_arr_cal = pol.bandpass_correct(stokes_arr, fn_bandpass)
        print("Calibrating xy correlation with %s\n\n" % src_linpol)
        stokes_arr_cal = pol.xy_correct(stokes_arr_cal, fn_xy_phase, 
                                    plot=inputs.mk_plot, clean=True)

        ivar = np.var(stokes_arr_cal[0],axis=1)[None,:,None]
        ivar = ivar**-1
        ivar[ivar==np.inf] = 0
        plot_dedisp(stokes_arr_cal*ivar, pulse_sample=pulse_sample, 
                    pulse_width=width_max, stokes=inputs.stokes, 
                    params=event_params, outdir=plotdir)

    if inputs.mk_plot:
        plot_all(stokes_arr, suptitle='Uncalibrated', 
                 fds=inputs.freq_downsample, tds=inputs.time_downsample,
                 stokes=inputs.stokes, params=event_params, outdir=plotdir)
        # mk_pol_plot(stokes_arr.reshape(4, 1536//16, 16, -1).mean(2),
        #         pulse_sample=pulse_sample, pulse_width=8)
        try:
           stokes_arr_cal
           plot_all(stokes_arr_cal, suptitle='xy-Calibrated', 
                    fds=inputs.freq_downsample, tds=inputs.time_downsample,
                    stokes=inputs.stokes, params=event_params, outdir=plotdir)
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
                     fds=inputs.freq_downsample, 
                     tds=inputs.time_downsample,
                     stokes=inputs.stokes, params=event_params, outdir=plotdir)

        if not inputs.polcal:
            Ucal, Vcal, xy_cal, phi_xy = pol.derotate_UV(stokes_vec[2], 
                    stokes_vec[3])
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

        print("Masking out:")
        print(mask)
        print(stokes_vec.shape)
        results_faraday = pol.faraday_fit(stokes_vec, RMmin=inputs.rmmin, 
                RMmax=inputs.rmmax, nrm=1000, nphi=200, 
                mask=mask, plot=inputs.mk_plot)
        RMs, P_derot_arr, RMmax, phimax, derot_phase = results_faraday

        if inputs.mk_plot:
            print("Attemping to plot")
            plot_RMspectrum(RMs, P_derot_arr, RMmax, 
                            phimax, derot_phase, 
                            fn_fig='%s_RMspectrum.pdf' % obs_name,
                            params=event_params, 
                            outdir=plotdir)

        print('Maximum likelihood RM: %0.2f' % RMmax)

        Pcal = (stokes_arr[1]+1j*stokes_arr[2])*derot_phase[:, None]
        stokes_arr[1], stokes_arr[2] = Pcal.real, Pcal.imag

        if inputs.mk_plot:
            plot_all(stokes_arr, suptitle='Faraday derotated', 
                     fds=inputs.freq_downsample, tds=inputs.time_downsample,
                     stokes=inputs.stokes, params=event_params, outdir=plotdir)







