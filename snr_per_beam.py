import sys 

import numpy as np
import optparse 

import tools

if __name__=='__main__':

    parser = optparse.OptionParser(prog="snr_per_beam.py", \
                        version="", \
                        usage="%prog fndir [OPTIONS]", \
                        description="Get SNR for each beam (CB or SB)")
    parser.add_option('--sig_thresh', dest='sig_thresh', type='float', \
                        help="Only process events above >sig_thresh S/N" \
                                "(Default: 5.0)", default=5.0)
    parser.add_option('--sig_thresh_ref', dest='sig_thresh', type='float', \
                        help="Only process events above >sig_thresh S/N" \
                                "(Default: 5.0)", default=8.0)
    parser.add_option('--save_data', dest='save_data', action='store_true', \
                        help="save data", default=False)
    parser.add_option('--mk_plot', dest='mk_plot', action='store_true', \
                        help="make plot if True (default False)", default=False)
    parser.add_option('--dm_min', dest='dm_min', type='float',
                        help="", 
                        default=0.0)
    parser.add_option('--dm_max', dest='dm_max', type='float',
                        help="", 
                        default=np.inf)
    parser.add_option('--t_max', dest='t_max', type='float',
                        help="Only process first t_max seconds", 
                        default=np.inf)
    parser.add_option('--t_window', dest='t_window', type='float',
                        help="", 
                        default=0.1)
    parser.add_option('--sb_ref', dest='t_window', type=int,
                        help="", 
                        default=35)
    parser.add_option('--cb_ref', dest='t_window', type=int,
                        help="", 
                        default=0)

    options, args = parser.parse_args()
    fdir = args[0]
    
    snr_arr = tools.cb_snr(fdir, ncb=40, dm_min=options.dm_min, dm_max=options.dm_max, \
                           sig_thresh_ref=options.sig_thresh_ref, cb_ref=options.cb_ref, \
                           t_window=options.t_window, sb_ref=options.sb_ref, nsb=71, \
                           sig_thresh=options.sig_thresh, mk_plot=options.mk_plot, \
                           )
    if options.save_data:
        np.save('%s/snr_data_beams.npy' % fdir, snr_arr)
