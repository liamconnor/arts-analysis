"""
Script to retreive all CBXX.trigger files 
from the 40 nodes, with the option to search them for 
triggers around a certain DM (sys.argv[3])
"""

import os
import sys

import numpy as np

import tools

directory = sys.argv[1]
outdir = sys.argv[2]

try:
    dm0 = np.float(sys.argv[3])
except:
    dm0 = 0

for ii in range(40):
    os.system('scp arts0%0.2d:/data2/output/%s/amber/CB%0.2d.trigger %s' % (ii+1,directory,ii,outdir))
    if dm0>0:
        fn = outdir + 'CB%0.2d.trigger' % ii
        dm, sig, tt, downsample, beam = tools.read_singlepulse(fn, beam='all')
        ind = np.where(np.abs(dm-dm0)<10)[0]
        for jj in ind:
            print(fn)
            print("DM:%0.2f T:%0.2f S/N:%0.2f" % (dm[jj], tt[jj], sig[jj]))
        print("")
