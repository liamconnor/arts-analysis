#!/usr/bin/env python
from __future__ import print_function, division
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import time
import multiprocessing
import tqdm

NOF_SUBBANDS = 1536

# Read input file
def read_dada_file(fname, headersize=4096):
    fp = open(fname, "r")
    header = fp.read(headersize)
    data = np.fromfile(fp, dtype="int8")
    fp.close()

    return data

# Check if a string is a float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Check if a string is an integer
def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False    

def read_event_params(fname):
    fp = open(fname, "r")
    event_params={}
    for ii,line in enumerate(fp):
        if line[:5]=="EVENT" or line[:4]=="BEAM":
            line_list = line.split(' ')
            param = line_list[0]
            for kk in line_list[1:]:
                if kk=='' or kk=='\n':
                    continue
                else:
                    event_params[param] = np.float(kk)
        if ii>100:
            break
    return event_params

# Parse DADA header
def read_dada_header(fname, headersize=4096):
    fp = open(fname, "r")
    header_str = fp.read(headersize)
    fp.close()

    # Header keywords to read
    header_keywords = ["HDR_SIZE",
                       "FILE_SIZE",
                       "UTC_START",
                       "MJD_START",
                       "OBS_OFFSET",
                       "FREQ",
                       "BW",
                       "NCHAN",
                       "TSAMP",
                       "NBIT",
                       "NDIM",
                       "NPOL",
                       "BYTES_PER_SECOND"]

    header = {}
    items = header_str.split()

    # Loop over items
    for i, item in enumerate(items):
        if item in header_keywords:
            if is_int(items[i+1]):
                header[item.lower()] = int(items[i+1])
            elif is_float(items[i+1]):
                header[item.lower()] = float(items[i+1])
            else:
                header[item.lower()] = items[i+1]            
    
    return header


def process(dada_filename, outdir='./'):

    print("Read data from", dada_filename)
    data = read_dada_file(dada_filename, headersize=hdr["hdr_size"])

    print('data.shape={}'.format(data.shape))
    
    
    #print("Selecting IQUV data")
    
    # fill_ringbuffer fill steps
    # package size => 8000 = 500 samples x 4 channels  x 4 (iquv)
    # curr_channel steps with increments of 4 => 200000 = 25 x 8000
    # tab_index steps with 1 => 776800000 = 1 x 1536/4 x 200000
    # data for 12 tabs = 12 x 776800000 = 921600000  ( this is 1 second )
    # seconds, tabs, nchannels/4, sequence_length, n_seq_numbers, c0-c3, iquv
    
    # data devided by data_sec_size must be an integer.
    #data_sec_size = 12 * 1536 * 12500 * 4
    data_sec_size = 12 * 384 * 25 * 500 * 4 * 4
    n_secs_max = int(data.shape[0] / data_sec_size)
    if args.nsec:
        if args.nsec > n_secs_max:
            print("Warning: setting n_sec to length of file: {}".format(n_secs_max))
        else:
            n_secs = args.nsec
    else:
        n_secs = n_secs_max

    useful_size = n_secs * data_sec_size 
    data = data[:useful_size]
    
    _data = data.reshape((-1, 12, 384, 12500, 4, 4))
    shape = _data.shape

    print(shape)

    n_samples = shape[0] * 12500

    # reorder data from 12 * 384 * 25 * 500 * 4 * 4
    print('reordening data can take a while')
    try:
        all_data = np.zeros((12, n_samples, 1536, 4))
        print('Data fit in memory')
    except MemoryError:
        print('Data do not fit in memory, using memmap')
        fname = 'memmap.dat'
        if os.path.isfile(fname):
            os.remove(fname)
        all_data = np.memmap(fname, dtype=int, mode='w+', shape=(12, n_samples, 1536, 4))
    #for sec in tqdm.tqdm(range(shape[0])):
    for sec in range(shape[0]):
        #for ch in tqdm.tqdm(range(0, 1536, 4)):
        for ch in range(0, 1536, 4):
            sample = sec * 12500
            sample_end = sample + 12500
            ch4 = int(ch / 4)
            ch_end = int(ch + 4)
            all_data[:,sample:sample_end,ch:ch_end,:] = _data[sec,:,ch4,:,:,:]

        
    iquv = {'I': (3, 'uint8'), 'Q': (2, 'int8'), 'U': (1, 'int8'), 'V': (0, 'int8')}

    # save data
    if args.tab is not None:
        tabs = [args.tab]
    else:
        tabs = range(12)
    if args.iquv is not None:
        keys = [args.iquv]
    else:
        keys = iquv.keys()
    for tab in tabs:
        for stokes in keys:
            print("TAB{:02d} Stokes {}".format(tab, stokes))
            iquv_nr, iquv_type = iquv[stokes]
            # save freq-time array
            np.save(outdir+"stokes{}_tab{:02d}".format(stokes, tab), all_data[tab, :, :, iquv_nr].astype(iquv_type).T)

    #for stokes_param in args.iquv:
        #stokes_param_to_plot = stokes_param.upper()
        #print('-----------------------------------------')
        #print('calculate statistics and plot data for {}'.format(stokes_param_to_plot))
        #print('-----------------------------------------')
        #iquv_nr = iquv[stokes_param_to_plot][0]
        #iquv_type = iquv[stokes_param_to_plot][1]
        #data_p = all_data[0,::-1,:,iquv_nr].astype(iquv_type)

        #np.save("stokes{}".format(stokes_param_to_plot), all_data[0, :, :, iquv_nr].astype(iquv_type))
        
        #try:
        #    signal = np.where(data_p != 0)
        #    data_s = data_p[signal]
        #    if data_s.shape[0] == 0:
        #        print('Only zeros in array')
        #        continue
        #    print('== statistics ==') 
        #    print('mean  = {}'.format(data_s.mean()))
        #    print('median= {}'.format(np.median(data_s)))
        #    print('max   = {}'.format(data_s.max()))
        #    print('min   = {}'.format(data_s.min()))
        #    #
        #    print('nonzero = {}%'.format(np.product(data_s.shape)*100./np.product(data_p.shape)))

        #    
        #    # Create plot
        #    fig = plt.figure(figsize=(20,11))
        #    shape = data_p.shape
        #    # print('shape={}'.format(shape))
        #    extent = [0, shape[1], shape[0], 0]
        #    min_val = np.min(data_p)
        #    #min_val = 0
        #    max_val = np.max(data_p)
        #    #max_val = 5
        #    #plt.imshow(data_p, aspect="auto", extent=extent, interpolation="hanning", origin="lower", vmin=-1, vmax=1)
        #    plt.imshow(data_p, aspect="auto", extent=extent, interpolation="nearest", origin="lower", vmin=min_val, vmax=max_val)
        #    fig.suptitle('filename= {},   plotted stokes parameter= {}'.format(dada_filename.split('/')[-1], stokes_param_to_plot))
        #    plt.ylabel("Sample number (12500/sec)")
        #    plt.xlabel("Frequency (channel)")
        #    plt.colorbar()
        #    #plot_filename = dada_filename.split("/")[5].replace('.dada',postfix)
        #    #print("Saving", plot_filename)
        #    #fig.savefig(plot_filename)
        #    fig.savefig("stokes{}.pdf".format(stokes_param_to_plot), bbox_inches='tight')
        #    #plt.show()
        #except ValueError:
        #    print('Only zeros in array')



if __name__ == "__main__":

    # We want to know how much time every step takes
    time_start = time.time()

    # Read command line arguments
    parser = argparse.ArgumentParser(description="Plot ARTS SC4 IQUV data")
    parser.add_argument("filename", help="dada file (base) name.")
    parser.add_argument("--iquv", nargs='*', type=str, help="stokes parameter to store [I | Q | U |V] (Default: all)")
    parser.add_argument("--nsec", type=int, help="Number of pages to process")
    parser.add_argument("--tab", type=int, help="Which TAB to process (Default: all)")
    parser.add_argument("--outdir", type=str, default='./', help="Output directory")
    args = parser.parse_args()

    dada_filename = args.filename
    hdr = read_dada_header(dada_filename)
    print(hdr)
    process(dada_filename, outdir=args.outdir)

    time_read_data_done = time.time()
    print("Time to read & process data:", time_read_data_done - time_start, "seconds")

    time_save_plot_done = time.time()
    print("Time to create and save plot(s):", time_save_plot_done - time_read_data_done, "seconds")





