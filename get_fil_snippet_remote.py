import os
import socket
import optparse

fnscript='/home/arts/connor/software/arts-analysis/arts-analysis/get_fil_snippet.py'
ncb=40

if __name__=='__main__':
    parser = optparse.OptionParser(prog="get_fil_snippet",
                                   version="",
                                   usage="%prog FN_FILTERBANK_PREFIX [OPTIONS]",
                                   description="Create diagnostic plots for individual triggers")


    parser.add_option('--outdir', dest='outdir', type='str',
                      help="directory to write data to on the node",
                      default='./data/')

    parser.add_option('--collectdir', dest='collectdir', type='str',
                      help="directory on arts041 to copy the data to",
                      default='')

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

    forked_jobs = []
    os.system('on_all_nodes.sh "mkdir -p %s"' % options.outdir) # in case it does not exist yet
    for ii in range(ncb):
#        print("Forking node " + str(ii+1))
#        os.system('sleep 5')
        child = os.fork()
        if child: # have the child, i.e. be the parent
            forked_jobs.append(child)
        else:
            # create the TAB snippet(s)
            ansible_args = (ii+1, fnscript, fn_fil, ii, ii, options.t, 
                            options.T, options.beams, options.central_freq,
                            options.outdir)
            os.system('ansible "arts0%0.2d.apertif" -a "nice python %s %s/CB%0.2d --cb %d --t %f --T %f --beams %s --central_freq %f --outdir %s"' % ansible_args)
            # copy the data to the collectdir
            if options.collectdir:
                if (socket.gethostname() != 'arts041'): 
                    print("You should probably not run this on a host different than arts041! Collect_dir may be incorrect")
                else: 
#                   os.system('ansible "arts0%0.2d.apertif" -a "cd %s; ionice scp CB%0.2d*.fil arts041:%s/"' % (ii+1, options.outdir, ii, options.collectdir))               
                    os.system('mkdir -p %s ; cd %s ; scp arts0%0.2d.apertif:%s/CB%0.2d*.fil .' % (options.collectdir, options.collectdir, ii+1, options.outdir, ii))
            os._exit(os.EX_OK) # killing this child   

    for child in forked_jobs:
        os.waitpid(child, 0)
