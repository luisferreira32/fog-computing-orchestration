#!/usr/bin/env python

# just a simple simulation

from sim_env.configs import SIM_TIME_STEPS
from sim_env.envrionment import Fog_env

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C

import numpy as np

env = DummyVecEnv([lambda: Fog_env()]) 
algorithm = PPO2(MlpPolicy, env)
obs = env.reset()
done = False; delays = []; discarded = 0
while not done:
	action, _states = algorithm.predict(obs)
	obs, rw, done, info = env.step(action)
	env.render()
	# info un vectorize
	info = info[0]
	delays = np.append(delays, info["delay_list"])
	discarded += info["discarded"]
print("delay_avg:",sum(delays)/len(delays),"processed:",len(delays),"discarded:",discarded)