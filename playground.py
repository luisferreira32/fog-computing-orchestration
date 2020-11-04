#!/usr/bin/env python

# just a simple simulation

from sim_env.configs import TRAINING_STEPS, N_NODES, DEFAULT_SLICES, RANDOM_SEED, ALGORITHM_SEED
from sim_env.envrionment import Fog_env
from algorithms.basic import Nearest_Round_Robin

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C

import numpy as np
import sys

if len(sys.argv) < 2:
	sys.argv.append("ppo2")
	sys.argv.append("a2c")
	sys.argv.append("rr")

display = False
if "-d" in sys.argv:
	display = True

if "ppo2" in sys.argv:
	env = DummyVecEnv([lambda: Fog_env()])
	# PPO2 test
	algorithm = PPO2(MlpPolicy, env, seed=ALGORITHM_SEED ,n_cpu_tf_sess=1)  
	algorithm.learn(total_timesteps=TRAINING_STEPS)
	obs = env.reset()
	done = False; ppo_delays = []; ppo_discarded = 0
	while not done:
		action, _states = algorithm.predict(obs)
		obs, rw, done, info = env.step(action)
		if display: env.render()
		# info gathering
		ppo_delays = np.append(ppo_delays, info[0]["delay_list"])
		ppo_discarded += info[0]["discarded"]

if "a2c" in sys.argv:
	env = DummyVecEnv([lambda: Fog_env()])
	#A2C
	algorithm = A2C(MlpPolicy, env, gamma=0.5, seed=ALGORITHM_SEED ,n_cpu_tf_sess=1)
	algorithm.learn(total_timesteps=TRAINING_STEPS)
	obs = env.reset()
	done = False; a2c_delays = []; a2c_discarded = 0
	while not done:
		action, _states = algorithm.predict(obs)
		obs, rw, done, info = env.step(action)
		if display: env.render()
		# info gathering
		a2c_delays = np.append(a2c_delays, info[0]["delay_list"])
		a2c_discarded += info[0]["discarded"]


if "rr" in sys.argv:
	env = DummyVecEnv([lambda: Fog_env()])
	#A2C
	algorithm = Nearest_Round_Robin(env.envs[0])
	obs = env.reset()
	done = False; rr_delays = []; rr_discarded = 0
	while not done:
		action = algorithm.predict(obs[0])
		obs, rw, done, info = env.step([action])
		if display: env.render()
		# info gathering
		rr_delays = np.append(rr_delays, info[0]["delay_list"])
		rr_discarded += info[0]["discarded"]

# result prints
if "ppo2" in sys.argv:
	print("ppo2 delay_avg:",sum(ppo_delays)/len(ppo_delays),"processed:",len(ppo_delays),"discarded:",ppo_discarded)
if "a2c" in sys.argv:
	print("a2c delay_avg:",sum(a2c_delays)/len(a2c_delays),"processed:",len(a2c_delays),"discarded:",a2c_discarded)
if "rr" in sys.argv:
	print("rr delay_avg:",sum(rr_delays)/len(rr_delays),"processed:",len(rr_delays),"discarded:",rr_discarded)