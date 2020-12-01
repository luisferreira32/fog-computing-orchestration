#!/usr/bin/env python

import time
import numpy as np

# for information gathering
from utils.tools import dictionary_append, append_to_file
from utils.display import info_gather, info_logs

def run_basic_algorithm_on_envrionment(agents, env, case, compiled_info=None, debug=False):
	# runner for simple baseline algorithms - they tweak inside the events => very model based of them!
	start_time = time.time()
	obs_n = env.reset()
	done = False;
	while not done:
		# for process they configure the actual EVQ!!!  => very model based of them! (but js)
		action_n = np.array([agent(obs, env.evq, env.clock) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
		obs_n, rw_n, done, info_n = env.step(action_n)
		if debug: env.render()
		# -- info gathering
		if compiled_info is not None: compiled_info = info_gather(compiled_info, info_n)
		# --

	# -- info logs
	if compiled_info is not None: info_logs(str(agents[0])+str(case), round(time.time()-start_time,2), compiled_info)
	# --
	return compiled_info


def run_algorithm_on_envrionment(agents, env, case, compiled_info=None, debug=False):
	# runner for trained algorithms
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