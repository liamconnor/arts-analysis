import os
import sys

import threading
import glob

def execute_amber(fn, nbatch=10800, hdr=460, 
                  rfi_option="-rfim", snr="mom_sigmacut", snrmin=6,
                  nchan=1536, pagesize=12500, chan_width=0.1953125, 
                  min_freq=1249.700927734375, tsamp=8.192e-05, output_prefix="./"):
    if hdr!=460:
        print("Using unconventional header length: %d" % hdr)

    if snr == "momad":
        conf_dir = "/home/oostrum/tuning/tuning_survey/momad/amber_conf"

        str_args_snr = (conf_dir, conf_dir, conf_dir, conf_dir)
        snr="-snr_momad -max_file %s/max.conf \
            -mom_stepone_file %s/mom_stepone.conf -mom_steptwo_file \
            %s/mom_steptwo.conf -momad_file %s/momad.conf" % str_args_snr
    elif snr == "mom_sigmacut":
        conf_dir = "/home/oostrum/tuning/tuning_survey/mom_sigmacut/amber_conf"
#        conf_dir = "/home/arts/ARTS-obs/amber_conf/"
        str_args_snr = (conf_dir, conf_dir, conf_dir)
        snr="-snr_mom_sigmacut -max_std_file %s/max_std.conf -mom_stepone_file %s/mom_stepone.conf -mom_steptwo_file %s/mom_steptwo.conf" % str_args_snr
    else:
        print("Unknown SNR mode: %s" % snr)

    str_args_general = (conf_dir, conf_dir, conf_dir, conf_dir, conf_dir, snrmin, conf_dir, conf_dir, conf_dir)
    general="amber -opencl_platform 0 -sync -print -padding_file %s/padding.conf -zapped_channels %s/zapped_channels_1400.conf \
    -integration_file %s/integration.conf -subband_dedispersion -dedispersion_stepone_file %s/dedispersion_stepone.conf \
    -dedispersion_steptwo_file %s/dedispersion_steptwo.conf -threshold %d -time_domain_sigma_cut \
    -time_domain_sigma_cut_steps %s/tdsc_steps.conf -time_domain_sigma_cut_configuration %s/tdsc.conf \
    -downsampling_configuration %s/downsampling.conf -compact_results" % str_args_general

    str_args_fil = (hdr, fn, nbatch, chan_width, min_freq, nchan, pagesize, tsamp)
    fil="-sigproc -stream -header %d -data %s -batches %d \
    -channel_bandwidth %f -min_freq %f -channels %d \
    -samples %d -sampling_time %0.5e" % str_args_fil

    str_args_step1 = (general, rfi_option, snr, fil, conf_dir, output_prefix)
    amber_step1="%s %s %s %s -opencl_device 1 -device_name ARTS_step1_81.92us_1400MHz -integration_steps %s/integration_steps_x1.conf \
    -subbands 32 -dms 32 -dm_first 0 -dm_step 0.2 -subbanding_dms 64 -subbanding_dm_first 0 -subbanding_dm_step 6.4 -output %s_step1" % str_args_step1

    str_args_step2 = (general, rfi_option, snr, fil, conf_dir, output_prefix)
    amber_step2="%s %s %s %s -opencl_device 2 -device_name ARTS_step2_81.92us_1400MHz \
    -integration_steps %s/integration_steps_x1.conf -subbands 32 -dms 32 -dm_first 0 -dm_step 0.2 -subbanding_dms 64 \
    -subbanding_dm_first 409.6 -subbanding_dm_step 6.4 -output %s_step2" % str_args_step2

    str_args_step3 = (general, rfi_option, snr, fil, conf_dir, output_prefix)
    amber_step3="%s %s %s %s -opencl_device 3 -device_name ARTS_step3_nodownsamp_81.92us_1400MHz \
    -integration_steps %s/integration_steps_x1.conf -subbands 32 -dms 16 -dm_first 0 -dm_step 2.5 \
    -subbanding_dms 64 -subbanding_dm_first 819.2 -subbanding_dm_step 40.0 -output %s_step3" % str_args_step3

    thread_step1 = threading.Thread(target=os.system, args=[amber_step1])
    thread_step2 = threading.Thread(target=os.system, args=[amber_step2])
    thread_step3 = threading.Thread(target=os.system, args=[amber_step3])

    threads = []
    thread_step1.daemon = True
    thread_step2.daemon = True
    thread_step3.daemon = True

    threads.append(thread_step1)
    threads.append(thread_step2)
    threads.append(thread_step3)
    
    thread_step1.start()
    thread_step2.start()
    thread_step3.start()

    # wait until all are done
    for thread in threads:
        thread.join()

    print("Done")
    return 

def run_amber_from_dir(dir, nbatch=1000, hdr=362,
                      rfi_option="-rfim", snr="mom_sigmacut", snrmin=5,
                      nchan=1536, pagesize=12500, chan_width=0.1953125,
                      min_freq=1249.700927734375, tsamp=8.192e-05):

    if dir.endswith('.fil'):
        files = [dir]
    else:
        files = glob.glob(dir+'*.fil')
    files.sort()

    outdir = './'

    for fn in files:
        outfn = outdir + fn.split('/')[-1].strip('.fil') + 'amber'
        execute_amber(fn, nbatch=nbatch, hdr=hdr,
                      rfi_option="-rfim", snr="mom_sigmacut", snrmin=5,
                      nchan=nchan, pagesize=pagesize, chan_width=chan_width,
                      min_freq=min_freq, tsamp=tsamp, output_prefix=outfn)

        os.system('cat %s*step*.trigger > %s.trigger' % (outfn, outfn))

    return '%s.trigger' % outfn

dir = sys.argv[1]
run_amber_from_dir(dir)






