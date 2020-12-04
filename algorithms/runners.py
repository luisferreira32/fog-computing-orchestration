#!/usr/bin/env python

import time
import numpy as np
import tensorflow as tf

# for information gathering
from utils.tools import dictionary_append, append_to_file
from utils.display import info_gather, info_logs
# the envrionment
from algorithms.deep_tools.common import  set_tf_seed
from algorithms.deep_tools.trainners import train_agents_on_env
from algorithms.deep_tools.savers import fetch_agents
from algorithms.configs import ALGORITHM_SEED


def run_basic_algorithm_on_envrionment(agents, env, case, compiled_info=None, debug=False):
	# runner for simple baseline algorithms
	start_time = time.time()
	obs_n = env.reset()
	done = False;
	while not done:
		action_n = np.array([agent.act(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
		obs_n, rw_n, done, info_n = env.step(action_n)
		if debug: env.render()
		# -- info gathering
		if compiled_info is not None: compiled_info = info_gather(compiled_info, info_n)
		# --

	# -- info logs
	if compiled_info is not None: info_logs(str(agents[0])+str(case), round(time.time()-start_time,2), compiled_info)
	# --
	return compiled_info


def run_rl_algorithm_on_envrionment(alg, env, case, compiled_info=None, debug=False, train=False):
	# runner for rl algorithms
	set_tf_seed(ALGORITHM_SEED)
	agents = fetch_agents(alg, env)

	# train them if requested
	if train: agents = train_agents_on_env(agents, env)

	# and run as usual
	start_time = time.time()
	obs_n = env.reset()
	done = False;
	while not done:
		action_n = np.array([agent.act(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
		obs_n, rw_n, done, info_n = env.step(action_n)
		if debug: env.render()
		# -- info gathering
		if compiled_info is not None: compiled_info = info_gather(compiled_info, info_n)
		# --

	# -- info logs
	if compiled_info is not None: info_logs(str(agents[0])+str(case), round(time.time()-start_time,2), compiled_info)
	# --
	# to clear all the models, since they're already saved after trainning
	tf.keras.backend.clear_session()
	return compiled_info