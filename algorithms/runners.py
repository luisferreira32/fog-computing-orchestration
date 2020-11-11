#!/usr/bin/env python

from utils.tools import dictionary_append, append_to_file
from utils.display import info_gather, info_logs

from sim_env.envrionment import Fog_env

import numpy as np
import time

def run_algorithm_on_envrionment(alg, case, seed, compiled_info, debug=False):
	start_time = time.time()
	env = Fog_env(case, seed)
	agents = [alg(n) for n in env.nodes]
	obs_n = env.reset()
	done = False;
	while not done:
		action_n = np.array([agent.decide(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
		obs_n, rw_n, done, info_n = env.step(action_n)
		if debug: env.render()
		# -- info gathering
		compiled_info = info_gather(compiled_info, info_n)
		# --

	# -- info logs
	info_logs(str(agents[0])+str(case), round(time.time()-start_time,2), compiled_info)
	# --
	return compiled_info