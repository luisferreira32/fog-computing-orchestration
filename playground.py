#!/usr/bin/env python

# THIS FILE IS NOT DOCUMENTED - USED AS A MAIN.PY ~ BUT JUST FOR BASIC EXAMPELS, IF YOU WANT TO RUN MORE THINGS MODIFY IT

# ---- JUST ESSENTIAL IMPORTS ----

import sys
import os
import time
from utils.input_handler import argument_check

# ---- command line input ----

[debug, algs, cases, max_seed, train, save] = argument_check(sys.argv)
if not algs or not cases:
	sys.exit(1)

# --- ALGORITHM AND ENVRIONMENT IMPORTS ---

from utils.tools import dictionary_append, write_all_to_csvs, random_seed_primes
from utils.display import plt_bar, plt_error_bar, plt_box_plot, info_gather_init
from sim_env.environment import Fog_env
from algorithms.runners import run_rl_algorithm_on_envrionment, run_basic_algorithm_on_envrionment

# --- the main playground ---

# info variables
delays_df={}; success_rate_df={}; overflow_rate_df={};
random_seeds = random_seed_primes(max_seed)
if not random_seeds: random_seeds = [2]


# ---- algorithms runnning for every case - GRAPHICAL RESULTS ----

print("[LOG] Running",len(algs),"algorithms for",len(cases),"cases with",len(random_seeds),"different seeds")
total = str(len(algs)*len(cases)*len(random_seeds)); current = 0
o_start_time = time.time()

# get some simulations to do the average
for case in cases:
	for seed in random_seeds:
		# generate the env
		env = Fog_env(case, seed)
		for alg in algs:
			start_time = time.time()
			# run either a basic algorithm or a RL algorithm
			if alg.basic:
				compiled_info = run_basic_algorithm_on_envrionment(alg, env, case, info_gather_init(), debug)
			else:
				compiled_info = run_rl_algorithm_on_envrionment(alg, env, case, info_gather_init(), debug, train, save)
			# run the algorithm to collect info
			# just to know
			current+=1; print("[LOG] simulations ran ["+str(current)+"/"+total+"] in",round(time.time()-o_start_time,2),"seconds")
			# for further use
			delays_df = dictionary_append(delays_df, case["case"]+alg.short_str(), compiled_info["average_delay"])
			success_rate_df = dictionary_append(success_rate_df, case["case"]+alg.short_str(), compiled_info["success_rate"])
			overflow_rate_df = dictionary_append(overflow_rate_df, case["case"]+alg.short_str(), compiled_info["overflow_rate"])


write_all_to_csvs(delays_df, success_rate_df, overflow_rate_df)
plt_bar(delays_df, mili=True, title="average_delays")
plt_error_bar(success_rate_df, mili=False, title="average_success_rate")
plt_error_bar(overflow_rate_df, mili=False, title="average_overflow_rate")
#plt_box_plot(delays_df, mili=True, title="average_delays")
