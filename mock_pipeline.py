import os 

import numpy as np
import glob 
import optparse

import run_amber_args 
import triggers 


if __name__=='__main__':
    parser = optparse.OptionParser(prog="tools.py", version="", usage="%prog fn1 fn2 [OPTIONS]", description="Compare to single-pulse trigger files")
    options, args = parser.parse_args()
    fdir = args[0]
    files = glob.glob(fdir)
    fnmod = '/home/arts/connor/software/single_pulse_ml_AA-ALERT/single_pulse_ml/single_pulse_ml/model/20190501freq_time.hdf5'

    for fn in files:
        fntrig = run_amber_args.run_amber_from_dir(fn, nbatch=10800, hdr=362,
                      rfi_option="-rfim", snr="mom_sigmacut", snrmin=5,
                      nchan=1536, pagesize=12500, chan_width=0.1953125,
                      min_freq=1249.700927734375, tsamp=8.192e-05)

        outdir = "/".join(fn.split('/'))
        print(outdir)
        os.system('python triggers.py --rficlean --mk_plot --save_data concat --descending_snr --sig_thresh 8.0 --ndm 1 --nfreq_plot 128 --ntime_plot 250 --outdir %s %s %s' % (outdir, fn_1, fntrig))

        fnh5 = outdir + './data/data_00_full.hdf5'

        os.system('cd /home/arts/connor/software/single_pulse_ml_AA-ALERT/single_pulse_ml/single_pulse_ml')
        os.system('tf')
        os.system('python classify.py %s %s --save_ranked --plot_ranked' % (fnh5, fnmod))