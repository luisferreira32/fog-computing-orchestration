#!/usr/bin/env python

# ---- JUST ESSENTIAL IMPORTS ----

import sys
from utils.input_handler import argument_check
from utils.basic_info import average_delays

# ---- command line input ----

[debug, algs, cases, max_seed] = argument_check(sys.argv)
if not algs or not cases:
	sys.exit(1)

# --- ALGORITHM AND ENVRIONMENT IMPORTS ---

from utils.tools import dictionary_append, write_all_to_csvs, random_seed_primes
from utils.display import plt_bar, plt_error_bar, plt_box_plot, info_gather_init
from sim_env.envrionment import Fog_env
from algorithms.basic import create_basic_agents
from algorithms.runners import run_algorithm_on_envrionment, create_and_train_agents

# --- the main playground ---

# info variables
delays_df={}; success_rate_df={}; overflow_rate_df={};
random_seeds = random_seed_primes(max_seed)
if not random_seeds: random_seeds = [2]
print("[LOG] Running",len(algs),"algorithms for",len(cases),"cases with",len(random_seeds),"different seeds")
total = str(len(algs)*len(cases)*len(random_seeds)); current = 0

# ---- algorithms runnning for every case ----

# get some simulations to do the average
for case in cases:
	for alg in algs:
		# trainable agents
		env = Fog_env(case, 2)
		if not alg.basic: agents = create_and_train_agents(env, alg, case)
		for seed in random_seeds:
			# generate the env
			env = Fog_env(case, seed)
			# and the basic agents
			if alg.basic:
				agents = create_basic_agents(env, alg)
			else:
				for agent, act_space in zip(agents, env.action_space.nvec):
					agent.set_action_space(act_space)
				
			# run the algorithm to collect info
			compiled_info = run_algorithm_on_envrionment(agents, env, case, info_gather_init(), debug)
			# just to know
			current+=1; print("[LOG] simulations ran ["+str(current)+"/"+total+"]")
			# for further use
			delays_df = dictionary_append(delays_df, alg.short_str()+case["case"], compiled_info["average_delay"])
			success_rate_df = dictionary_append(success_rate_df, alg.short_str()+case["case"], compiled_info["success_rate"])
			overflow_rate_df = dictionary_append(overflow_rate_df, alg.short_str()+case["case"], compiled_info["overflow_rate"])


write_all_to_csvs(delays_df, success_rate_df, overflow_rate_df)
plt_bar(delays_df, mili=True, title="average_delays")
plt_error_bar(success_rate_df, mili=False, title="average_success_rate")
plt_error_bar(overflow_rate_df, mili=False, title="average_overflow_rate")
#plt_box_plot(delays_df, mili=True, title="average_delays")
