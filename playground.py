#!/usr/bin/env python

# just a simple simulation

from sim_env.configs import TRAINING_STEPS, N_NODES, DEFAULT_SLICES
from sim_env.envrionment import Fog_env

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C

import numpy as np
import sys

env = DummyVecEnv([lambda: Fog_env()])

if len(sys.argv) > 1 and sys.argv[1] == "seed":
	# test reproductability
	algorithm = PPO2(MlpPolicy, env)
	# an action for doing nothing
	obs = env.reset()
	a, _ = algorithm.predict(obs)
	obs, _, _, _ = env.step(a)
	print(obs)
	a, _ = algorithm.predict(obs)
	obs, _, _, _ = env.step(a)
	print(obs)
	obs = env.reset()
	a, _ = algorithm.predict(obs)
	obs, _, _, _ = env.step(a)
	print(obs)
	a, _ = algorithm.predict(obs)
	obs, _, _, _ = env.step(a)
	print(obs)

if len(sys.argv) > 1 and "ppo2" in sys.argv:
	# PPO2 test
	algorithm = PPO2(MlpPolicy, env)  
	#algorithm.learn(total_timesteps=TRAINING_STEPS)
	obs = env.reset()
	done = False; ppo_delays = []; ppo_discarded = 0
	while not done:
		action, _states = algorithm.predict(obs)
		obs, rw, done, info = env.step(action)
		env.render()
		# info gathering
		ppo_delays = np.append(ppo_delays, info[0]["delay_list"])
		ppo_discarded += info[0]["discarded"]

if len(sys.argv) > 1 and "a2c" in sys.argv:
	#A2C
	algorithm = A2C(MlpPolicy, env, gamma=0.5)
	algorithm.learn(total_timesteps=TRAINING_STEPS)
	obs = env.reset()
	done = False; a2c_delays = []; a2c_discarded = 0
	while not done:
		action, _states = algorithm.predict(obs)
		obs, rw, done, info = env.step(action)
		# env.render()
		# info gathering
		a2c_delays = np.append(a2c_delays, info[0]["delay_list"])
		a2c_discarded += info[0]["discarded"]

# result prints
if "ppo2" in sys.argv:
	print("ppo2 delay_avg:",sum(ppo_delays)/len(ppo_delays),"processed:",len(ppo_delays),"discarded:",ppo_discarded)
if "a2c" in sys.argv:
	print("a2c delay_avg:",sum(a2c_delays)/len(a2c_delays),"processed:",len(a2c_delays),"discarded:",a2c_discarded)