import os
import sys

import glob

files = glob.glob('/home/oostrum/debug/full_output_TAB/filterbank/*.fil')
files = glob.glob('../../injectfrb/injectfrb/data/simpulse_nfrb10_DM50-50_61sec_20190415-0839.fil')
files = glob.glob(sys.argv[1])
files.sort()

#script = '/data1/rfi-tests-peryton/run_amber.sh'
outdir = './dany_tests/'
outdir = './'
script = '/home/arts/connor/software/arts-analysis/arts-analysis/run_amber.sh'

combis = [('-rfim','momad'),('','momad'),('-rfim','mom_sigmacut'),('','mom_sigmacut')]
combis = [('','mom_sigmacut')]

for fn in files[:]:
    for algo in combis[:]:
        rfi = algo[0]
        snr = algo[1]
        outfn = outdir + fn.split('/')[-1].strip('.fil') + 'amber_tester%s_%s' % (rfi, snr)
                
        print('%s %s %s %s %s' % (script, fn, rfi, snr, outfn))

        os.system('%s %s %s %s %s' % (script, fn, snr, outfn, rfi))
        os.system('cat %s*step*.trigger > %s.trigger' % (outfn, outfn))


#!/bin/bash

def execute_amber(file, nbatch=10800, hdr=362, 
				  rfi_option="-rfim", snr="mom_sigmacut", snrmin=6,
				  nchan=1536, pagesize=12500, chan_width=0.1953125, 
				  min_freq=1249.700927734375, tsamp=8.192e-05, output_prefix="./"):
	if hdr!=460:
		print("Using unconventional header length: %d" % hdr)

	if snr == "momad":
	    conf_dir = "/home/oostrum/tuning/tuning_survey/momad/amber_conf"
	    snr="-snr_momad -max_file $conf_dir/max.conf -mom_stepone_file $conf_dir/mom_stepone.conf -mom_steptwo_file $conf_dir/mom_steptwo.conf -momad_file $conf_dir/momad.conf"
	elif snr == "mom_sigmacut":
	    conf_dir = "/home/oostrum/tuning/tuning_survey/mom_sigmacut/amber_conf"
	    snr="-snr_mom_sigmacut -max_std_file $conf_dir/max_std.conf -mom_stepone_file $conf_dir/mom_stepone.conf -mom_steptwo_file $conf_dir/mom_steptwo.conf"
	else:
	    print("Unknown SNR mode: $snr")

	general="amber -opencl_platform 0 -sync -print -padding_file $conf_dir/padding.conf -zapped_channels $conf_dir/zapped_channels_1400.conf -integration_file $conf_dir/integration.conf -subband_dedispersion -dedispersion_stepone_file $conf_dir/dedispersion_stepone.conf -dedispersion_steptwo_file $conf_dir/dedispersion_steptwo.conf -threshold $snrmin -time_domain_sigma_cut -time_domain_sigma_cut_steps $conf_dir/tdsc_steps.conf -time_domain_sigma_cut_configuration $conf_dir/tdsc.conf -downsampling_configuration $conf_dir/downsampling.conf"

	fil="-sigproc -stream -header $hdr -data $file -batches $nbatch -channel_bandwidth $chan_width -min_freq $min_freq -channels $nchan -samples $pagesize -sampling_time $tsamp"

	str_args_step1 = (general, rfi_option, snr, fil, output_prefix)
	amber_step1="%s %s %s %s -opencl_device 1 \
				 -device_name ARTS_step1_81.92us_1400MHz \
				 -integration_steps $conf_dir/integration_steps_x1.conf \
				 -subbands 32 -dms 32 -dm_first 0 -dm_step 0.2 -subbanding_dms 64 \
				 -subbanding_dm_first 0 -subbanding_dm_step 6.4 \
				 -output %s_step1" % str_args_step1

	os.system(amber_step1)
	exit()
	amber_step2="$general $rfi_option $snr $fil -opencl_device 2 -device_name ARTS_step2_81.92us_1400MHz -integration_steps $conf_dir/integration_steps_x1.conf -subbands 32 -dms 32 -dm_first 0 -dm_step 0.2 -subbanding_dms 64 -subbanding_dm_first 409.6 -subbanding_dm_step 6.4 -output ${output_prefix}_step2"
	#amber_step3="$general $rfi_option $snr $fil -opencl_device 3 -device_name ARTS_step3_81.92us_1400MHz -integration_steps $conf_dir/integration_steps_x5.conf -subbands 32 -dms 32 -dm_first 0 -dm_step 0.5 -subbanding_dms 128 -subbanding_dm_first 819.2 -subbanding_dm_step 16.0 -output ${output_prefix}_step3 -downsampling -downsampling_factor 5"
	amber_step3="$general $rfi_option $snr $fil -opencl_device 3 -device_name ARTS_step3_nodownsamp_81.92us_1400MHz -integration_steps $conf_dir/integration_steps_x1.conf -subbands 32 -dms 16 -dm_first 0 -dm_step 2.5 -subbanding_dms 64 -subbanding_dm_first 819.2 -subbanding_dm_step 40.0 -output ${output_prefix}_step3"

	print("Starting AMBER on filterbank")
	#$amber_step1 &
	#$amber_step2 &
	#$amber_step3 &

	print("Done")



