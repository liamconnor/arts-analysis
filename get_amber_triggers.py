"""
Script to retreive all CBXX.trigger files 
from the 40 nodes, with the option to search them for 
triggers around a certain DM (sys.argv[3])
"""

import os
import sys

import numpy as np
import optparse

import tools

#if len(sys.argv) < 2:
#    print("Expecting, e.g. python get_amber_triggers.py 20190925/2019-09-25-22:50:52.FRB190709 /data1/data/FRBs/FRB190925/amber/ 956.0")
#    exit()

#directory = sys.argv[1]
#outdir = sys.argv[2]

try:
    dm0 = np.float(sys.argv[3])
except:
    dm0 = 0

try:
    t0_ = np.float(sys.argv[4])
except:
    t0_ = 0

try: 
    CBs = [np.int(sys.argv[5])]
except:
    CBs = range(40)

if __name__=='__main__':
    def foo_callback(option, opt, value, parser):
      setattr(parser.values, option.dest, value.split(','))

    parser = optparse.OptionParser(prog="snr_per_beam.py", \
                        version="", \
                        usage="%prog fndir [OPTIONS]", \
                        description="Get SNR for each beam (CB or SB)")
    parser.add_option('--sig_thresh', dest='sig_thresh', type='float', \
                        help="Only process events above >sig_thresh S/N" \
                                "(Default: 5.0)", default=5.0)
    parser.add_option('--t0', dest='t0', type='float', \
                        help="Arrival time of pulse", default=0.0)
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
    parser.add_option('--sb_ref', dest='sb_ref', type=int,
                        help="", 
                        default=None)
    parser.add_option('--cb_ref', dest='cb_ref', type=int,
                        help="", 
                        default=0)
    parser.add_option('--CBs',
                  type='string',
                  action='callback',
                  callback=foo_callback, 
                  )


    options, args = parser.parse_args() 

    directory = args[0]
    outdir = '/tmp/'
    dm_max = options.dm_max
    dm_min = options.dm_min 
    sig_thresh = options.sig_thresh
    CBs = range(40)

    print(dm_min, dm_max)
    for ii in CBs:
        os.system('scp arts0%0.2d:/data2/output/%s/amber/CB%0.2d.trigger %s' % (ii+1,directory,ii,outdir))
        print('scp arts0%0.2d:/data2/output/%s/amber/CB%0.2d.trigger %s' % (ii+1,directory,ii,outdir))
        if dm0>0:
            fn = outdir + 'CB%0.2d.trigger' % ii

            if not os.path.isfile(fn):
                continue

            dm, sig, tt, downsample, beam = tools.read_singlepulse(fn, beam='all')

            if len(dm)==0:
                print("ntohing")
                continue 

            if t0_==0:
                t0 = tt
            else:
                t0 = t0_

            ind = np.where((np.abs(dm-dm0)<5) & () (np.abs(tt-t0)<5.0))[0]
            ind = np.where((dm<dm_max) & (dm>dm_min) & (np.abs(tt-t0)<5.0) & (sig>sig_thresh))[0]    
            
            if len(ind)>0:
                print(fn)
            else:
                print("\nNo corresponding triggers\n")

            for jj in ind:
                print("DM:%0.2f SB:%d T:%0.2f S/N:%0.2f" % (dm[jj], beam[jj], tt[jj], sig[jj]))
            print("")
