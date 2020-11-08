#!/usr/bin/env python

# ---- JUST ESSENTIAL IMPORTS ----

import sys
from utils.input_handler import argument_check

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

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2 #, A2C

import numpy as np



# ---- algorithms runnning for every case ----
# info variables
delays_list=[]; success_rates_list = [];

# main loop
for case in cases:
	# --- PPO2 algorithm ---
	if "ppo2" in algs:
		env = DummyVecEnv([lambda: Fog_env(case)])
		# PPO2 test
		algorithm = PPO2(MlpPolicy, env, seed=ALGORITHM_SEED ,n_cpu_tf_sess=1, verbose=0)  
		algorithm.learn(total_timesteps=TRAINING_STEPS)
		obs = env.reset()
		done = False; delays = [];success_rates=[]; discarded = 0
		while not done:
			action, _states = algorithm.predict(obs)
			obs, rw, done, info = env.step(action)
			if debug: env.render()
			# info gathering
			info = info[0]
			delays = np.append(delays, info["delay_list"])
			success_rates = np.append(success_rates, info["success_rate"])
			discarded += info["discarded"]
		delays_list.append(delays)
		success_rates_list.append(success_rates)

	# --- Nearest Node - Round Robin algorithm ---
	if "rr" in algs:
		env = DummyVecEnv([lambda: Fog_env(case)])
		algorithm = Nearest_Round_Robin(env.envs[0])
		obs = env.reset()
		done = False; delays = []; success_rates=[]; rr_discarded = 0
		while not done:
			action = algorithm.predict(obs[0])
			obs, rw, done, info = env.step([action])
			if debug: env.render()
			# info gathering
			info = info[0]
			delays = np.append(delays, info["delay_list"])
			success_rates = np.append(success_rates, info["success_rate"])
			rr_discarded += info["discarded"]
		delays_list.append(delays)
		success_rates_list.append(success_rates)

	# --- Nearest Node - Priority Queue algorithm --- 
	if "pq" in algs:
		env = DummyVecEnv([lambda: Fog_env(case)])
		algorithm = Nearest_Priority_Queue(env.envs[0])
		obs = env.reset()
		done = False; delays = [];success_rates=[]; pq_discarded = 0
		while not done:
			action = algorithm.predict(obs[0])
			obs, rw, done, info = env.step([action])
			if debug: env.render()
			# info gathering
			info = info[0]
			delays = np.append(delays, info["delay_list"])
			success_rates = np.append(success_rates, info["success_rate"])
			pq_discarded += info["discarded"]
		delays_list.append(delays)
		success_rates_list.append(success_rates)

x = [case["case"]+" "+alg for case in cases for alg in algs]
plt_error_bar(x, success_rates_list, title="success_rates")
plt_bar(x, [np.mean(d) for d in delays_list], mili=True, title="average_delays")

""" OBSOLETE CODE --- will delete in the future
if "a2c" in algs:
	env = DummyVecEnv([lambda: Fog_env()])
	#A2C
	algorithm = A2C(MlpPolicy, env, gamma=0.5, seed=ALGORITHM_SEED ,n_cpu_tf_sess=1)
	algorithm.learn(total_timesteps=TRAINING_STEPS)
	obs = env.reset()
	done = False; a2c_delays = []; a2c_discarded = 0
	while not done:
		action, _states = algorithm.predict(obs)
		obs, rw, done, info = env.step(action)
		if debug: env.render()
		# info gathering
		a2c_delays = np.append(a2c_delays, info[0]["delay_list"])
		a2c_discarded += info[0]["discarded"]


# result prints
if "ppo2" in algs:
	print("ppo2 delay_avg:",sum(ppo_delays)/len(ppo_delays),"processed:",len(ppo_delays),"discarded:",ppo_discarded)
if "a2c" in algs:
	print("a2c delay_avg:",sum(a2c_delays)/len(a2c_delays),"processed:",len(a2c_delays),"discarded:",a2c_discarded)
if "rr" in algs:
	print("rr delay_avg:",sum(rr_delays)/len(rr_delays),"processed:",len(rr_delays),"discarded:",rr_discarded)
if "pq" in algs:
	print("pq delay_avg:",sum(pq_delays)/len(pq_delays),"processed:",len(pq_delays),"discarded:",pq_discarded)
"""