"""
Script to retreive all CBXX.trigger files 
from the 40 nodes, with the option to search them for 
triggers around a certain DM (sys.argv[3])
"""

import os
import sys

import numpy as np
import optparse
import matplotlib.pylab as plt

import tools

# ARTS-discovered FRBs
known_dms_apertif = [663., 956.7, 465.0, 587.0, 531.0]
name_apertif = ['Ap1','Ap2','Ap3','Ap4','Ap5']
repeater_dms = [560.0, 189.0, 349.2, 103.5, 450.5, 364.05, 441.0, 1281.6, 425.0, 460.6, 
                580.05, 552.65, 302, 195.6, 393.6, 222.4, 1378.2, 651.45, 309.6]
name = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10', 
        'R190208','R190604','R190212','R180908','R190117',
        'R190303','R190417','R190213','R190907']

repeater_dm_list = known_dms_apertif + repeater_dms
name_list = name_apertif + name

for ii in range(len(repeater_dm_list)):
    print("%s : %0.2f" % (name_list[ii], repeater_dm_list[ii]))

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
    parser.add_option('--dm0', dest='dm0', type='float',
                        help="", 
                        default=0.0)
    parser.add_option('--dm_min', dest='dm_min', type='float',
                        help="", 
                        default=0.0)
    parser.add_option('--dm_max', dest='dm_max', type='float',
                        help="", 
                        default=np.inf)
    parser.add_option('--mk_plot', dest='mk_plot', action='store_true', \
                        help="make plot if True (default False)", default=False)
    parser.add_option('--outdir', dest='outdir', type='str',
                        help="", 
                        default='/tmp/')
    parser.add_option('--wf_command', dest='wf_command', action='store_true', \
                        help="print command line for waterfaller", default=False)
    parser.add_option('--CBs',
                  type='string',
                  action='callback',
                  callback=foo_callback, 
                  )


    options, args = parser.parse_args() 

    if args[0].endswith('.trigger'):
        fn = args[0]
        do_scp = False
    else:
        do_scp = True

    directory = args[0]
    outdir = options.outdir
    dm_max = options.dm_max
    dm_min = options.dm_min 
    sig_thresh = options.sig_thresh
    dm0 = options.dm0

    if options.CBs is None:
        CBs = range(40)
    else:
        CBs = options.CBs

    if dm0!=0:
        dm_min = dm0 - 5.0
        dm_max = dm0 + 5.0
    if np.abs(dm_min-dm_max)>100:
        print("DM range will produce too many triggers, changing to 0--100")
        dm_min = 50.
        dm_max = 150.

    dm0 = 0.5*(dm_min+dm_max)

    for ii in CBs:
        ii = int(ii)

        if do_scp:
            os.system('scp arts0%0.2d:/data2/output/%s/amber/CB%0.2d.trigger %s' % (ii+1,directory,ii,outdir))
#            os.system('echo scp arts0{:0.2d}:/data2/output/%s/amber/CB%0.2d.trigger %s' % (ii+1,directory,ii,outdir))y
            fn = outdir + 'CB%0.2d.trigger' % ii

        if not os.path.isfile(fn):
            print("Input file does not exist")
            continue

        dm, sig, tt, downsample, beam = tools.read_singlepulse(fn, beam='all')

        if len(dm)==0:
            print("nothing")
            continue 

        if options.mk_plot is True:
            fig = plt.figure()
            plt.scatter(tt, dm, 2*sig, color='k', alpha=0.35)
            plt.axhline(dm0, color='red', alpha=0.25)
            for jj, dms in enumerate(known_dms_apertif):
                plt.axhline(dms, color='green', alpha=0.25, linestyle='--')
                plt.text(1.05*tt.max(), dms, name_apertif[jj], fontsize=6)
            for jj, dms in enumerate(repeater_dms):
                plt.axhline(dms, color='blue', alpha=0.25, linestyle='--')
                plt.text(1.05*tt.max(), dms, name[jj], fontsize=6)

            plt.xlabel('Time [s]', fontsize=16)
            plt.ylabel('DM', fontsize=16)

            plt.title('%s\nCB%0.2d' % (directory, ii), fontsize=15)
            plt.show()

        if options.t0==0:
            t0 = tt 
        else:
            t0 = options.t0

        ind = np.where((dm<dm_max) & (dm>dm_min) & (np.abs(tt-t0)<5.0) & (sig>sig_thresh))[0]    
        
        if len(ind)>0:
            print(fn)
        else:
            print("\nNo corresponding triggers\n")

        for jj in ind[:100]:
            str_arg1 = (dm[jj], beam[jj], tt[jj], downsample[jj], sig[jj])
            print("DM:%0.2f SB:%d T:%0.2f W:%d S/N:%0.2f" % str_arg1)
            if options.wf_command:
                fnfil = '/data2/output/' + directory + '/filterbank/CB%.2d' % ii
                str_arg2 = (dm[jj], beam[jj], tt[jj], downsample[jj], fnfil)
                print("python waterfall_sb.py --dm %0.2f --sb %d --t %0.2f --downsamp %d %s\n" % str_arg2)







