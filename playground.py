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

from sim_env.configs import TOTAL_TIME_STEPS, N_NODES, DEFAULT_SLICES, RANDOM_SEED_LIST
from sim_env.envrionment import Fog_env

from algorithms.configs import ALGORITHM_SEED
from algorithms.basic import Nearest_Round_Robin, Nearest_Priority_Queue

from utils.tools import dictionary_append, append_to_file, random_seed_primes
from utils.display import plt_bar, plt_error_bar, plt_box_plot
from utils.display import info_gather_init, info_gather, info_logs

import numpy as np
import time


# ---- algorithms runnning for every case ----
# info variables
delays_df={}; success_rate_df={}; overflow_rate_df={};
random_seeds = random_seed_primes(200) # primes from 0 - N

# main loop
for case in cases:

	# --- baselines for a multi-agent gym ---
	# get some simulations to do the average
	for seed in RANDOM_SEED_LIST:
		# --- Nearest Node - Round Robin algorithm ---
		if "rr" in algs:
			# -- information related
			start_time = time.time()
			compiled_info = info_gather_init()
			# --
			env = Fog_env(case, seed)
			agents = [Nearest_Round_Robin(n) for n in env.nodes]
			obs_n = env.reset()
			done = False;
			while not done:
				action_n = np.array([agent.decide(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
				obs_n, rw_n, done, info_n = env.step(action_n)
				if debug: env.render()
				# -- info gathering
				compiled_info = info_gather(compiled_info, info_n)
				# --

			# -- info logs
			info_logs(str(agents[0])+str(case), round(time.time()-start_time,2), compiled_info)
			# for further use
			delays_df = dictionary_append(delays_df, "rr"+case["case"], compiled_info["average_delay"])
			success_rate_df = dictionary_append(success_rate_df, "rr"+case["case"], compiled_info["success_rate"])
			overflow_rate_df = dictionary_append(overflow_rate_df, "rr"+case["case"], compiled_info["overflow_rate"])
			# --
		print(success_rate_df)

		
	for seed in RANDOM_SEED_LIST:
		# --- Nearest Node - Priority Queue algorithm --- 
		if "pq" in algs:
			# -- information related
			start_time = time.time()
			compiled_info = info_gather_init()
			# --
			env = Fog_env(case, seed)
			agents = [Nearest_Priority_Queue(n) for n in env.nodes]
			obs_n = env.reset()
			done = False; delays = []; overflowed=[0 for n in env.nodes]; discarded=[0 for n in env.nodes];
			while not done:
				action_n = np.array([agent.decide(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
				obs_n, rw_n, done, info_n = env.step(action_n)
				if debug: env.render()
				# -- info gathering
				compiled_info = info_gather(compiled_info, info_n)
				# --

			# -- info logs
			info_logs(str(agents[0])+str(case), round(time.time()-start_time,2), compiled_info)
			# for further use
			delays_df = dictionary_append(delays_df, "pq"+case["case"], compiled_info["average_delay"])
			success_rate_df = dictionary_append(success_rate_df, "pq"+case["case"], compiled_info["success_rate"])
			overflow_rate_df = dictionary_append(overflow_rate_df, "pq"+case["case"], compiled_info["overflow_rate"])
			# --


plt_bar(delays_df, mili=True, title="average_delays")
plt_error_bar(success_rate_df, mili=False, title="average_success_rate")
#plt_box_plot(delays_df, mili=True, title="average_delays")
