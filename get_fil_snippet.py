"""
author: liam connor 
date : 27 September 2019
Script to cut out data from filterbank file, save
and to a new filterbank file.
"""

import os

import optparse
import numpy as np

import reader 
import triggers_liam as triggers

sb_generator = triggers.SBGenerator.from_science_case(science_case=4)
sb_generator.reversed = True
ntab=12

def save_fil_snippet(fn_fil, options, sb=None, sb_gen=None, tabno=-1, dt=8.192e-5):
    ntime_plot = np.int(options.T / 8.192e-5)
    print(fn_fil)
    x = triggers.proc_trigger(fn_fil, 1e-5, options.t, -1,
                 ndm=1, mk_plot=False, downsamp=1,
                 beamno=options.cb, fn_mask=None, nfreq_plot=1536,
                 ntime_plot=ntime_plot,
                 rficlean=False, snr_comparison=-1,
                 outdir=options.outdir, sig_thresh_local=0.0,
                 subtract_zerodm=False,
                 n_iter_time=3, n_iter_frequency=3, clean_type='time', freq=options.central_freq,
                 sb_generator=sb_generator, sb=sb, save_snippet=True, tabno=tabno)

def setup_save(fn_fil, options):
    if options.beams in ['alltabs', 'tabs', 'ALLTABS', 'allTABs']:
        sb = None
        tabs = range(ntab)
        for tab in tabs:
            print(tab)
            save_fil_snippet(fn_fil+'_%0.2d.fil' % tab, options, sb=None, sb_gen=None, tabno=tab)
        exit()

    if options.beams in ['allsbs', 'sbs', 'sb', 'allSB', 'allsb']:
        sbs = range(70)
        triggers.mpl.use('Agg', warn=False)
        import matplotlib.pyplot as plt
    else:
        reload(triggers.mpl)
        triggers.mpl.use('TkAgg', warn=False)
        import matplotlib.pyplot as plt
        reload(plt)
        try:
            sbs = [int(options.beams)]
        except:
            print("Don't know which beams you want")
            exit()

    print(sbs)
    for sb in sbs:
        print('Plotting SB %s' % sb)
        save_fil_snippet(fn_fil, options, sb=sb, sb_gen=sb_generator, tabno=None)


if __name__=='__main__':
    parser = optparse.OptionParser(prog="get_fil_snippet",
                                   version="",
                                   usage="%prog FN_FILTERBANK_PREFIX [OPTIONS]",
                                   description="Create diagnostic plots for individual triggers")

    parser.add_option('--outdir', dest='outdir', type='str',
                      help="directory to write data to",
                      default='./data/')

    parser.add_option('--T', dest='T', type='float',
                      help="time of data snippet in seconds",
                      default=10.0)

    parser.add_option('--t', dest='t', type='float',
                      help="Arrival time of pulse in seconds (default 10)",
                      default=10.0)

    parser.add_option('--cb', dest='cb', type='int',
                      help="compound beam",
                      default=-1)

    parser.add_option('--beams', dest='beams', type=str, default='tabs',
                      help="Process synthesized beams")

    parser.add_option('--central_freq', dest='central_freq', type=float, default=1370.0, 
                      help="Central frequency in zapped channels filename (Default: 1370)")

    options, args = parser.parse_args()
    fn_fil = args[0]

    setup_save(fn_fil, options)

    exit()


#os.sytem('ansible "arts001.apertif,arts002.apertif" -a "command"')
