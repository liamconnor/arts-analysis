import os

import optparse

fnscript='/home/arts/connor/software/arts-analysis/arts-analysis/get_fil_snippet.py'
ncb=40

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

    for ii in range(ncb):
        ansible_args = (ii+1, fnscript, fn_fil, ii, options.t, 
                        options.T, options.beams, options.central_freq,
                        options.outdir)
        os.system('ansible "arts0%0.2d.apertif" -a "python %s %s/CB%0.2d --t %f --T %f --beams %s --central_freq %f --outdir %s"' % ansible_args)
