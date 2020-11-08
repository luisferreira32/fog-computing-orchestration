#!/usr/bin/env python

# necessary constants import
from sim_env.configs import BASE_SLICE_CHARS
from sim_env.configs import NORMAL_CASE_1, NORMAL_CASE_2, NORMAL_CASE_3
from sim_env.configs import HEAVY_CASE_1, HEAVY_CASE_2, HEAVY_CASE_3

def argument_check(argv):
	# running variables
	debug = False
	algs = []; cases = []

	# argument check
	if len(argv) < 2:
		print("run with --help or -H for more information")
		return [debug, algs, cases]

	# help print
	if "--help" in argv or "-H" in argv:
		print("Playground.py should be played in an envrionment with Tensorflow and OpenAI gym")
		print("--basic [-B] : runs a basic run")
		print("--algorithm= [-A=] : choose your algorithm, by default rr is chosen")
		print("   rr : Nearest Round Robin basline algorithm")
		print("   pq : Nearest Priority Queue basline algorithm")
		print("--cases= [-C=] : by default runs case 1 with normal traffic")
		print("   all : runs case 1, 2 and 3, with normal and heavy traffic")
		print("   normal : runs case 1, 2 and 3, with normal traffic")
		print("   heavy : runs case 1, 2 and 3, with heavy traffic")
		print("   n1 : runs case X =[1, 2 or 3] with normal traffic")
		print("   h1 : runs case X =[1, 2 or 3] with heavy traffic")
		print("--debug : will render every step")
		return [debug, algs, cases]

	# pick up the flags
	if "--debug" in argv:
		debug = True

	for s in argv:
		if "--algorithm=" in s or "-A=" in s:
			if "rr" in s:
				algs.append("rr")
			if "pq" in s:
				algs.append("pq")
		if "--cases=" in s or "-C=" in s:
			cases = []
			if "all" in s:
				cases = [NORMAL_CASE_1, NORMAL_CASE_2, NORMAL_CASE_3, HEAVY_CASE_1, HEAVY_CASE_2, HEAVY_CASE_3]
			elif "normal" in s:
				cases = [NORMAL_CASE_1, NORMAL_CASE_2, NORMAL_CASE_3]
			elif "heavy" in s:
				cases = [HEAVY_CASE_1, HEAVY_CASE_2, HEAVY_CASE_3]
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
			algs.append("rr")
			cases = [BASE_SLICE_CHARS]

	# default values if it was not chosen
	if not algs:
		algs = ["rr"]
	if not cases:
		cases = [BASE_SLICE_CHARS]

	return [debug, algs, cases]