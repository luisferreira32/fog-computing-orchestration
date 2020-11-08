#!/usr/bin/env python

# ---- JUST ESSENTIAL IMPORTS ----

import sys
from utils.input_handler import argument_check
from utils.basic_info import average_delays

# ---- command line input ----

[debug, algs, cases] = argument_check(sys.argv)
if not algs or not cases:
	sys.exit(1)

# --- ALGORITHM IMPORTS ---

from sim_env.configs import TRAINING_STEPS, N_NODES, DEFAULT_SLICES, RANDOM_SEED
from algorithms.configs import ALGORITHM_SEED
from sim_env.envrionment import Fog_env
from algorithms.basic import Nearest_Round_Robin, Nearest_Priority_Queue
from utils.display import plt_bar, plt_error_bar

import numpy as np



# ---- algorithms runnning for every case ----
# info variables
delays_list=[]; success_rates_list = [];

# main loop
for case in cases:

	# --- baselines for a multi-agent gym ---

	# --- Nearest Node - Round Robin algorithm ---
	if "rr" in algs:
		env = Fog_env(case)
		average_delays(env.nodes)
		agents = [Nearest_Round_Robin(n) for n in env.nodes]
		obs_n = env.reset()
		done = False; delays = []; success_rates=[]; rr_discarded = 0
		while not done:
			action_n = np.array([agent.decide(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
			obs_n, rw_n, done, info_n = env.step(action_n)
			if debug: env.render()
			# info gathering
			for key, info in info_n.items():
				delays = np.append(delays, info["delay_list"])
		print("Nearest_Round_Robin", round(1000*sum(delays)/len(delays),2),"ms on case",case)

	# --- Nearest Node - Priority Queue algorithm --- 
	if "pq" in algs:
		env = Fog_env(case)
		agents = [Nearest_Priority_Queue(n) for n in env.nodes]
		obs_n = env.reset()
		done = False; delays = []; success_rates=[]; rr_discarded = 0
		while not done:
			action_n = np.array([agent.decide(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
			obs_n, rw_n, done, info = env.step(action_n)
			# info gathering
			#... here ....


#x = [case["case"]+" "+alg for case in cases for alg in algs]
#plt_bar(x, [np.mean(d) for d in delays_list], mili=True, title="average_delays")
#plt_error_bar(x, success_rates_list, title="success_rates")
