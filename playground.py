#!/usr/bin/env python

# ---- JUST ESSENTIAL IMPORTS ----

import sys
import os
import time
from utils.input_handler import argument_check, fetch_trained_agents
from utils.basic_info import average_delays

# ---- command line input ----

[debug, algs, cases, max_seed, train] = argument_check(sys.argv)
if not algs or not cases:
	sys.exit(1)

# --- ALGORITHM AND ENVRIONMENT IMPORTS ---

from utils.tools import dictionary_append, write_all_to_csvs, random_seed_primes
from utils.display import plt_bar, plt_error_bar, plt_box_plot, info_gather_init
from sim_env.envrionment import Fog_env
from algorithms.basic import create_basic_agents
from algorithms.runners import run_algorithm_on_envrionment
from algorithms.deep_tools.frames import Conv1d_Frame, Rnn_Frame, Conv1d_Rnn_Frame

# --- the main playground ---

# info variables
delays_df={}; success_rate_df={}; overflow_rate_df={};
random_seeds = random_seed_primes(max_seed)
if not random_seeds: random_seeds = [2]

models_linux_path = os.getcwd() + "/algorithms/saved_models/"
models_windows_path = os.getcwd() + "\\algorithms\\saved_models\\"

# ---- training ONE algorithm for ONE case ----
if train:
	alg = algs[0]
	case = cases[0]
	n_episodes = 100
	env = Fog_env(case, 2)
	print("[LOG] Training",alg.short_str(),"in case",case["case"],"for",n_episodes,"episodes")
	agents = [alg(n.index) for n in env.nodes]
	agents = alg.train_agents_on_env(agents, env, n_episodes)
	my_path = models_windows_path+alg.short_str()+case["case"]+"\\"
	for agent in agents:
		agent.save_models(my_path)
	sys.exit()
# ---- algorithms runnning for every case - GRAPHICAL RESULTS ----

print("[LOG] Running",len(algs),"algorithms for",len(cases),"cases with",len(random_seeds),"different seeds")
total = str(len(algs)*len(cases)*len(random_seeds)); current = 0
o_start_time = time.time()

# get some simulations to do the average
for case in cases:
	for alg in algs:
		for seed in random_seeds:
			start_time = time.time()
			# generate the env
			env = Fog_env(case, seed)
			# and the basic agents
			if alg.basic:
				agents = create_basic_agents(env, alg, case)
			else:
				agents = fetch_trained_agents(env, alg, case)
			if not agents:
				break
			# run the algorithm to collect info
			compiled_info = run_algorithm_on_envrionment(agents, env, case, info_gather_init(), debug)
			# just to know
			current+=1; print("[LOG] simulations ran ["+str(current)+"/"+total+"] in",round(time.time()-start_time,2),"seconds [total:",round(time.time()-o_start_time,2),"s]")
			# for further use
			delays_df = dictionary_append(delays_df, case["case"]+alg.short_str(), compiled_info["average_delay"])
			success_rate_df = dictionary_append(success_rate_df, case["case"]+alg.short_str(), compiled_info["success_rate"])
			overflow_rate_df = dictionary_append(overflow_rate_df, case["case"]+alg.short_str(), compiled_info["overflow_rate"])


write_all_to_csvs(delays_df, success_rate_df, overflow_rate_df)
plt_bar(delays_df, mili=True, title="average_delays")
plt_error_bar(success_rate_df, mili=False, title="average_success_rate")
plt_error_bar(overflow_rate_df, mili=False, title="average_overflow_rate")
#plt_box_plot(delays_df, mili=True, title="average_delays")
