import os

import numpy as np
import glob 

from darc.sb_generator import SBGenerator

ntab = 12
nfreq = 1536
ncb = 40
nsb = 71

sb_generator = SBGenerator.from_science_case(science_case=4)
sb_generator.reversed = True

# Directory with drift scan .npy files.
fdir='/home/arts/leon/20190709/data/'
model_outdir = '/data1/data/FRBs/FRB190709/beammodel/'
cb_model = '/home/arts/connor/software/arts-localisation/models/all_cb.npy'
data_full_sb = []
ntime_list = []

def build_model():
    for ii in range(ncb):
        print("CB%0.2d" % ii)
        fl = glob.glob(fdir + 'CB%0.2d*.npy' % ii)
        fl.sort()
        data_full = []                      
        for fn in fl:
            print(fn)                                                                 
            data = np.load(fn)                         
            data = data-np.median(data, axis=-1, keepdims=True)
            data_full.append(data)                     
        try:
            data_full = np.concatenate(data_full).reshape(ntab, nfreq, -1)        
        except:
            print("\nConcat didn't work for CB%d\n" % ii)                                                               
            print(data.shape)                                                                                       
            [data_full_sb.append(np.zeros([nfreq, 1000])) for xx in range(nsb)]
            continue
        for sb in range(nsb):
            D = sb_generator.synthesize_beam(data_full, sb)
            data_full_sb.append(D)

    return data_full_sb

    ntime_tot = 500
    ntime_min = np.inf
    for beam_arr in data_full_sb:
        ntime = beam_arr.shape[-1]
        if ntime < ntime_min:
            ntime_min = ntime 

    ntime_transit_cbs = []
    for ii in range(ncb):
        if data_full_sb[nsb*ii + nsb//2].sum()==0:
            ntime_transit_cbs.append(ntime_tot//2)
            continue
        central_sb = data_full_sb[nsb*ii + nsb//2]
        ntime_transit = np.argmax(central_sb.mean(0))
        ntime_transit_cbs.append(ntime_transit)

    data_full_sb_clipped = np.empty([ncb, nsb, nfreq, ntime_tot])

    for ii in range(ncb):
        for jj in range(nsb):
            data = data_full_sb[nsb*ii + jj]
            data = data[:, ntime_transit_cbs[ii]-ntime_tot//2:ntime_transit_cbs[ii]+ntime_tot//2]
            if data.shape[-1] != ntime_tot:
                print(ii, jj, data.shape)
                continue

            data_full_sb_clipped[ii, jj] = data 

    del data_full_sb
    return data_full_sb_clipped

fnout_mod = model_outdir+'3C84_sb_beam_model.npy'

if os.path.exists(fnout_mod):
    data_full_sb_clipped = np.load(fnout_mod)
else:
    data_full_sb_clipped = build_model()
    data_full_sb_clipped = data_full_sb_clipped.reshape(ncb, nsb, nfreq//16, 16, -1).mean(-2)
    np.save(model_outdir+'3C84_sb_beam_model.npy', data_full_sb_clipped)

cb_arr = np.load(cb_model)
ntheta, nphi = cb_arr.shape[1], cb_arr.shape[2]
#sb_arr = np.empty([ncb, nsb, ntheta, nphi])

for ii in range(ncb):
    print("Needs a fix: Beams are in wrong location... Need to be re-centered")
    mmax = np.argmax(cb_arr[ii].mean(0))
    cb_arr_ii = cb_arr[ii] * np.ones([nsb, 1, 1])
    cb_arr_ii[..., mmax-250:mmax+250] *= data_full_sb_clipped[ii].mean(-2)[:, None]
#    sb_arr = cb_arr[ii, None, :, mmax-250:mmax+250] * data_full_sb_clipped[ii].mean(-2)[:, None]
#    sb_arr = cb_arr[ii, None, :, :1250] * data_full_sb_clipped[ii].mean(-2)[:, None]
#    np.save(model_outdir+'sb_model_3C48_%02d' % ii, sb_arr)













