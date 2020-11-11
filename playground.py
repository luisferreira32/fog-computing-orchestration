#!/usr/bin/env python

# ---- JUST ESSENTIAL IMPORTS ----

import sys
from utils.input_handler import argument_check
from utils.basic_info import average_delays

# ---- command line input ----

[debug, algs, cases] = argument_check(sys.argv)
if not algs or not cases:
	sys.exit(1)

# --- ALGORITHM AND ENVRIONMENT IMPORTS ---

from utils.tools import dictionary_append, write_all_to_csvs, random_seed_primes
from utils.display import plt_bar, plt_error_bar, plt_box_plot, info_gather_init

from algorithms.runners import run_algorithm_on_envrionment

# --- the main playground ---

# info variables
delays_df={}; success_rate_df={}; overflow_rate_df={};
random_seeds = random_seed_primes(0) # 30: first 10 primes, 100: first 25 primes, 235: first 50 primes
if not random_seeds: random_seeds = [2]

# ---- algorithms runnning for every case ----

# get some simulations to do the average
for seed in random_seeds:
	for case in cases:
		for alg in algs:
			compiled_info = run_algorithm_on_envrionment(alg, case, seed, info_gather_init(), debug)
			# for further use
			delays_df = dictionary_append(delays_df, alg.short_str()+case["case"], compiled_info["average_delay"])
			success_rate_df = dictionary_append(success_rate_df, alg.short_str()+case["case"], compiled_info["success_rate"])
			overflow_rate_df = dictionary_append(overflow_rate_df, alg.short_str()+case["case"], compiled_info["overflow_rate"])


#write_all_to_csvs(delays_df, success_rate_df, overflow_rate_df)
plt_bar(delays_df, mili=True, title="average_delays")
plt_error_bar(success_rate_df, mili=False, title="average_success_rate")
plt_error_bar(overflow_rate_df, mili=False, title="average_overflow_rate")
#plt_box_plot(delays_df, mili=True, title="average_delays")
