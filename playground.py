#!/usr/bin/env python

# ---- JUST ESSENTIAL IMPORTS ----

import sys
from sim_env.configs import BASE_SLICE_CHARS
from sim_env.configs import NORMAL_CASE_A, NORMAL_CASE_B, NORMAL_CASE_C
from sim_env.configs import HEAVY_CASE_A, HEAVY_CASE_B, HEAVY_CASE_C

# ---- command line input ----

# argument check
if len(sys.argv) < 2:
	print("run with --help or -H for more information")
	sys.exit(1)

# help print
if "--help" in sys.argv or "-H" in sys.argv:
	print("Playground.py should be played in an envrionment with Tensorflow and OpenAI gym")
	print("--basic [-B] : runs a basic run")
	print("--algorithm= [-A=] : choose your algorithm, by default rr is chosen")
	print("   rr : Nearest Round Robin basline algorithm")
	print("   pq : Nearest Priority Queue basline algorithm")
	print("   ppo2 : needs Tensorflow 1.15 and stable-baslines")
	print("--cases= [-C=] : by default runs case A with normal traffic")
	print("   all : runs case A, B and C, with normal and heavy traffic")
	print("   normal : runs case A, B and C, with normal traffic")
	print("   heavy : runs case A, B and C, with heavy traffic")
	print("   _A : runs case A with normal and heavy traffic")
	print("   _B : runs case B with normal and heavy traffic")
	print("   _C : runs case C with normal and heavy traffic")
	print("--debug : will render every step")
	sys.exit(1)

# running variables
debug = False
algs = []; cases = []

# pick up the flags
if "--debug" in sys.argv:
	debug = True

for s in sys.argv:
	if "--algorithm=" in s or "-A=" in s:
		if "rr" in s:
			algs.append("rr")
		if "pq" in s:
			algs.append("pq")
		if "ppo2" in s:
			algs.append("ppo2")
	if "--cases=" in s or "-C=" in s:
		if "all" in s:
			cases = [NORMAL_CASE_A, NORMAL_CASE_B, NORMAL_CASE_C, HEAVY_CASE_A, HEAVY_CASE_B, HEAVY_CASE_C]
		elif "normal" in s:
			cases = [NORMAL_CASE_A, NORMAL_CASE_B, NORMAL_CASE_C]
		elif "heavy" in s:
			cases = [HEAVY_CASE_A, HEAVY_CASE_B, HEAVY_CASE_C]
		elif "_A" in s:
			cases = [NORMAL_CASE_A, HEAVY_CASE_A]
		elif "_B" in s:
			cases = [NORMAL_CASE_B, HEAVY_CASE_B]
		elif "_C" in s:
			cases = [NORMAL_CASE_C, HEAVY_CASE_C]
		elif "HC" in s:
			cases = [HEAVY_CASE_C]
		elif "NC" in s:
			cases = [NORMAL_CASE_C]
	if "--basic" in s or "-B" in s:
		algs.append("rr")
		cases = [BASE_SLICE_CHARS]

# default values if it was not chosen
if not algs:
	algs = ["rr"]
if not cases:
	cases = [BASE_SLICE_CHARS]


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
avg_delays_list = []; delays_list=[];
print(cases)
for case in cases:
	if "ppo2" in algs:
		env = DummyVecEnv([lambda: Fog_env(case)])
		# PPO2 test
		algorithm = PPO2(MlpPolicy, env, seed=ALGORITHM_SEED ,n_cpu_tf_sess=1, verbose=0)  
		algorithm.learn(total_timesteps=TRAINING_STEPS)
		obs = env.reset()
		done = False; delays = []; discarded = 0
		while not done:
			action, _states = algorithm.predict(obs)
			obs, rw, done, info = env.step(action)
			if debug: env.render()
			# info gathering
			delays = np.append(delays, info[0]["delay_list"])
			discarded += info[0]["discarded"]
		delays_list.append(delays)
		avg_delays_list.append(sum(delays)/len(delays))

	if "rr" in algs:
		env = DummyVecEnv([lambda: Fog_env(case)])
		algorithm = Nearest_Round_Robin(env.envs[0])
		obs = env.reset()
		done = False; delays = []; rr_discarded = 0
		while not done:
			action = algorithm.predict(obs[0])
			obs, rw, done, info = env.step([action])
			if debug: env.render()
			# info gathering
			delays = np.append(delays, info[0]["delay_list"])
			rr_discarded += info[0]["discarded"]
		delays_list.append(delays)
		avg_delays_list.append(sum(delays)/len(delays))

	if "pq" in algs:
		env = DummyVecEnv([lambda: Fog_env(case)])
		algorithm = Nearest_Priority_Queue(env.envs[0])
		obs = env.reset()
		done = False; delays = []; pq_discarded = 0
		while not done:
			action = algorithm.predict(obs[0])
			obs, rw, done, info = env.step([action])
			if debug: env.render()
			# info gathering
			delays = np.append(delays, info[0]["delay_list"])
			pq_discarded += info[0]["discarded"]
		delays_list.append(delays)
		avg_delays_list.append(sum(delays)/len(delays))

plt_error_bar(algs, delays_list)
#plt_bar(algs, avg_delays_list)

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