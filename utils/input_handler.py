#!/usr/bin/env python

# necessary constants import
from sim_env.configs import BASE_SLICE_CHARS
from sim_env.configs import NORMAL_CASES, HEAVY_CASES, ALL_CASES
from sim_env.configs import NORMAL_CASE_1, NORMAL_CASE_2, NORMAL_CASE_3
from sim_env.configs import HEAVY_CASE_1, HEAVY_CASE_2, HEAVY_CASE_3
from algorithms.configs import ALGORITHM_SEED
# algorithms
from algorithms.basic import Nearest_Round_Robin, Nearest_Priority_Queue
from algorithms.a2c import A2c_Orchestrator
from algorithms.dqn import Dqn_Orchestrator

def argument_check(argv):
	# running variables
	debug = False; train = False; save = False;
	algs = []; cases = []
	max_seed = 10

	# argument check
	if len(argv) < 2:
		print("run with --help or -H for more information")
		return [debug, algs, cases, max_seed, train, save]

	# help print
	if "--help" in argv or "-H" in argv:
		print("Playground.py should be played in an envrionment with Tensorflow and OpenAI gym")
		print("--basic [-B] : runs a basic run")
		print("--algorithm= [-A=] : choose your algorithm, by default rr is chosen")
		print("   rr : Nearest Round Robin basline algorithm")
		print("   pq : Nearest Priority Queue basline algorithm")
		print("   a2c : Advantage Actor Critic RL algorithm with PPO trainstep")
		print("   dqn : Deep Q-Networks RL algorithm")
		print("--cases= [-C=] : by default runs case 1 with normal traffic")
		print("   all : runs case 1, 2 and 3, with normal and heavy traffic")
		print("   normal : runs case 1, 2 and 3, with normal traffic")
		print("   heavy : runs case 1, 2 and 3, with heavy traffic")
		print("   n1 : runs case X =[1, 2 or 3] with normal traffic")
		print("   h1 : runs case X =[1, 2 or 3] with heavy traffic")
		print("--seedmax= : by default 10, maximum value for a prime number seed")
		print("--debug : will render every step")
		print("--train : will train the model before running sims")
		print("--save : will save the model")
		return [debug, algs, cases, max_seed, train, save]

	# pick up the flags
	if "--debug" in argv:
		debug = True

	for s in argv:
		if "--algorithm=" in s or "-A=" in s:
			if "pq" in s:
				algs.append(Nearest_Priority_Queue)
			if "rr" in s:
				algs.append(Nearest_Round_Robin)
			if "a2c" in s:
				algs.append(A2c_Orchestrator)
			if "dqn" in s:
				algs.append(Dqn_Orchestrator)
		if "--cases=" in s or "-C=" in s:
			cases = []
			if "all" in s:
				cases = ALL_CASES
			elif "normal" in s:
				cases = NORMAL_CASES
			elif "heavy" in s:
				cases = HEAVY_CASES
			# add ons	
			if "n1" in s:
				cases.append(NORMAL_CASE_1)
			if "n2" in s:
				cases.append(NORMAL_CASE_2)
			if "n3" in s:
				cases.append(NORMAL_CASE_3)
			if "h1" in s:
				cases.append(HEAVY_CASE_1)
			if "h2" in s:
				cases.append(HEAVY_CASE_2)
			if "h3" in s:
				cases.append(HEAVY_CASE_3)
		if "--basic" in s or "-B" in s:
			algs = [Nearest_Round_Robin]
			cases = [NORMAL_CASE_1]
			break # actually will just run default

		if "--seedmax=" in s:
			try:
				max_seed = int(s.replace("--seedmax=",''))
			except:
				max_seed = 10
		if "--train" in s:
			train = True
		if "--save" in s:
			save = True

	# default values if it was not chosen
	if not algs:
		algs = [Nearest_Round_Robin]
	if not cases:
		cases = [BASE_SLICE_CHARS]

	return [debug, algs, cases, max_seed, train, save]
