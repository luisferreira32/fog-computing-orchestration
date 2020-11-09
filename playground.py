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

from sim_env.configs import TOTAL_TIME_STEPS, N_NODES, DEFAULT_SLICES, RANDOM_SEED
from sim_env.envrionment import Fog_env

from algorithms.configs import ALGORITHM_SEED
from algorithms.basic import Nearest_Round_Robin, Nearest_Priority_Queue

from utils.tools import dictionary_append, append_to_file
from utils.display import plt_bar, plt_error_bar, plt_box_plot, info_logs

import numpy as np
import time


# ---- algorithms runnning for every case ----
# info variables
delays_df={};

# main loop
for case in cases:

	# --- baselines for a multi-agent gym ---

	# --- Nearest Node - Round Robin algorithm ---
	if "rr" in algs:
		env = Fog_env(case)
		agents = [Nearest_Round_Robin(n) for n in env.nodes]
		print("Started",agents[0],"with case",case); start_time = time.time()
		obs_n = env.reset()
		done = False; delays = []; overflowed=[0 for n in env.nodes]; discarded=[0 for n in env.nodes];
		while not done:
			action_n = np.array([agent.decide(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
			obs_n, rw_n, done, info_n = env.step(action_n)
			if debug: env.render()
			# info gathering
			for key, info in info_n.items():
				delays = np.append(delays, info["delay_list"])
				overflowed[key-1] += info["overflow"]
				discarded[key-1] += info["discarded"]

		# info logs
		info_logs(agents[0],case, round(time.time()-start_time,2), round(1000*sum(delays)/len(delays),2),
			round(len(delays)/(len(delays)+sum(overflowed)+sum(discarded)),2),
			round(sum(overflowed)/(len(delays)+sum(overflowed)+sum(discarded)),2))
		# for further use
		delays_df = dictionary_append(delays_df, "rr"+case["case"], delays)
		append_to_file("delays.txt",{"Round Robin "+case["case"]:delays})

	# --- Nearest Node - Priority Queue algorithm --- 
	if "pq" in algs:
		env = Fog_env(case)
		agents = [Nearest_Priority_Queue(n) for n in env.nodes]
		print("Started",agents[0],"with case",case); start_time = time.time()
		obs_n = env.reset()
		done = False; delays = []; overflowed=[0 for n in env.nodes]; discarded=[0 for n in env.nodes];
		while not done:
			action_n = np.array([agent.decide(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
			obs_n, rw_n, done, info_n = env.step(action_n)
			if debug: env.render()
			# info gathering
			for key, info in info_n.items():
				delays = np.append(delays, info["delay_list"])
				overflowed[key-1] += info["overflow"]
				discarded[key-1] += info["discarded"]

		# info logs
		info_logs(agents[0],case, round(time.time()-start_time,2), round(1000*sum(delays)/len(delays),2),
			round(len(delays)/(len(delays)+sum(overflowed)+sum(discarded)),2),
			round(sum(overflowed)/(len(delays)+sum(overflowed)+sum(discarded)),2))
		# for further use
		delays_df = dictionary_append(delays_df, "pq"+case["case"], delays)
		append_to_file("delays.txt",{"Priority Queue "+case["case"]:delays})


#plt_bar(delays_df, mili=True, title="average_delays")
#plt_error_bar(delays_df, mili=True, title="average_delays")
plt_box_plot(delays_df, mili=True, title="average_delays")
