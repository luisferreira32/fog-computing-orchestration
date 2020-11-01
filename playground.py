#!/usr/bin/env python

# just a simple simulation

from sim_env.configs import SIM_TIME_STEPS
from sim_env.envrionment import Fog_env

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C

env = DummyVecEnv([lambda: Fog_env()]) 
algorithm = PPO2(MlpPolicy, env)
obs = env.reset()
done = False
while not done:
	action, _states = algorithm.predict(obs)
	obs, rw, done, info = env.step(action)
	env.render()