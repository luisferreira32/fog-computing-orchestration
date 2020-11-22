#!/usr/bin/env python

# for information gathering
from utils.tools import dictionary_append, append_to_file
from utils.display import info_gather, info_logs

# the envrionment
from sim_env.envrionment import Fog_env

# mathematic and tensor related
import numpy as np
import time
import tensorflow as tf
import tqdm

# type setting
from typing import Any, List, Sequence, Tuple
# training agents stuff
from algorithms.deep_tools.trainers import train_actor_critic, set_training_env, run_episode

# -- baseline related runners --

def run_algorithm_on_envrionment(agents, env, case, compiled_info=None, debug=False):
	# runner for simple baseline algorithms
	start_time = time.time()
	obs_n = env.reset()
	done = False;
	while not done:
		action_n = np.array([agent(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
		obs_n, rw_n, done, info_n = env.step(action_n)
		if debug: env.render()
		# -- info gathering
		if compiled_info is not None: compiled_info = info_gather(compiled_info, info_n)
		# --

	# -- info logs
	if compiled_info is not None: info_logs(str(agents[0])+str(case), round(time.time()-start_time,2), compiled_info)
	# --
	return compiled_info

# to train RL agents  on an envrionment
def run_agents_on_env(env, agents, max_episodes: int = 100):
	running_reward = 0
	reward_threshold = 10000
	max_steps_per_episode = 1000
	# Run the model for one episode to collect training data
	with tqdm.trange(max_episodes) as t:
		for i in t:
			initial_state = set_training_env(env)
			episode_reward = train_actor_critic(initial_state, agents, gamma=0.9, max_steps=max_steps_per_episode)

			running_reward = sum(episode_reward)*0.01 + running_reward*.99

			t.set_description(f'Episode {i}')
			#t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
			print(running_reward)

			if running_reward > reward_threshold:  
				break
	return agents