#!/usr/bin/env python

# necessary constants import
from sim_env.configs import BASE_SLICE_CHARS
from sim_env.configs import NORMAL_CASE_A, NORMAL_CASE_B, NORMAL_CASE_C
from sim_env.configs import HEAVY_CASE_A, HEAVY_CASE_B, HEAVY_CASE_C

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
		print("   ppo2 : needs Tensorflow 1.15 and stable-baslines")
		print("--cases= [-C=] : by default runs case A with normal traffic")
		print("   all : runs case A, B and C, with normal and heavy traffic")
		print("   normal : runs case A, B and C, with normal traffic")
		print("   heavy : runs case A, B and C, with heavy traffic")
		print("   _A : runs case A with normal and heavy traffic")
		print("   _B : runs case B with normal and heavy traffic")
		print("   _C : runs case C with normal and heavy traffic")
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
			if "ppo2" in s:
				algs.append("ppo2")
		if "--cases=" in s or "-C=" in s:
			if "all" in s:
				cases = [NORMAL_CASE_A, NORMAL_CASE_B, NORMAL_CASE_C, HEAVY_CASE_A, HEAVY_CASE_B, HEAVY_CASE_C]
			elif "normal" in s:
				cases = [NORMAL_CASE_A, NORMAL_CASE_B, NORMAL_CASE_C]
			elif "heavy" in s:
				cases = [HEAVY_CASE_A, HEAVY_CASE_B, HEAVY_CASE_C]
			elif "_A" in s:
				cases = [NORMAL_CASE_A, HEAVY_CASE_A]
			elif "_B" in s:
				cases = [NORMAL_CASE_B, HEAVY_CASE_B]
			elif "_C" in s:
				cases = [NORMAL_CASE_C, HEAVY_CASE_C]
			elif "HC" in s:
				cases = [HEAVY_CASE_C]
			elif "NC" in s:
				cases = [NORMAL_CASE_C]
		if "--basic" in s or "-B" in s:
			algs.append("rr")
			cases = [BASE_SLICE_CHARS]

	# default values if it was not chosen
	if not algs:
		algs = ["rr"]
	if not cases:
		cases = [BASE_SLICE_CHARS]

	return [debug, algs, cases]