#!/usr/bin/env python

import numpy as np

from sim_env.configs import N_NODES, DEFAULT_SLICES, MAX_QUEUE, RAM_UNIT, TIME_STEP
from sim_env.events import Start_processing, Stop_processing
from utils.tools import uniform_rand_array

class Nearest_Round_Robin(object):
	"""Nearest_Round_Robin
	"""
	basic = True
	def __init__(self, node, case):
		# to keep track of what slice is processing
		self.process = 0
		self.node = node
		self.case = case

	def __str__(self):
		return "Nearest Round Robin"

	@staticmethod
	def short_str():
		return "nRR"

	def act(self, obs):
		# node action struct: [f_i0, ..., f_ik, w_i0, ..., w_ik]
		# default no offloads and no processing
		wks = np.zeros(DEFAULT_SLICES, dtype=np.uint8)
		fks = np.zeros(DEFAULT_SLICES, dtype=np.uint8)

		# for every node make an decision
		[a_k, b_k, be_k, rc_k, rm_k] = np.split(obs, [DEFAULT_SLICES, DEFAULT_SLICES*2, DEFAULT_SLICES*3, DEFAULT_SLICES*3+1])
		
		# to process based on availabe memory, cpu, and RR priority
		while sum(b_k) > sum(be_k) and rm_k >= np.ceil(self.case["task_type"][self.process][2]/RAM_UNIT) and rc_k > 0:
			# set the processing, w_ik
			wks[self.process] += 1
			# and take the resources on the available obs
			rm_k -= int(np.ceil(self.case["task_type"][self.process][2]/RAM_UNIT))
			rc_k -= 1
			be_k[self.process] += 1
			# next time step we'll process a different slice
			self.process +=1
			if self.process == DEFAULT_SLICES:
				self.process = 0

		# offload to the Nearest Node if buffer bigger than 0.8
		for k in range(self.node.max_k):
			if a_k[k] == 1:
				fks[k] = self.node.index
			if b_k[k] >= 0.8*MAX_QUEUE and a_k[k] == 1:
				# set the f_ik to the nearest node
				min_d = 10000; min_n = 0
				for n, d in self.node._distances.items():
					if d < min_d:
						min_n = n
						min_d = d
				fks[k] =min_n

		# and return the action
		return np.append(fks, wks)



class Nearest_Priority_Queue(object):
	"""Nearest_Priority_Queue
	"""
	basic = True
	def __init__(self, node, case):
		self.node = node
		self.case = case
		# make slice priority on nodes
		delay_constraints = [k[0] for k in case["task_type"]]
		# to make priorities shuffeled at equal values
		aux_random = uniform_rand_array(len(delay_constraints))
		self.priorities = np.lexsort((aux_random, delay_constraints))

	def __str__(self):
		return "Nearest Priority Queue"
		
	@staticmethod
	def short_str():
		return "nPQ"

	def act(self, obs):
		# action struct: [f_00, ..., f_0k, w_00, ..., w_0k, ..., f_i0, ..., f_ik, w_i0, ..., w_ik]
		# default no offloads and no processing
		wks = np.zeros(DEFAULT_SLICES, dtype=np.uint8)
		fks = np.zeros(DEFAULT_SLICES, dtype=np.uint8)

		# for every node make an decision
		[a_k, b_k, be_k, rc_k, rm_k] = np.split(obs, [DEFAULT_SLICES, DEFAULT_SLICES*2, DEFAULT_SLICES*3, DEFAULT_SLICES*3+1])
		
		# begining with the higher to the lower priorities (slice k)
		for k in self.priorities:
			# to process based on availabe memory and there is still tasks to process
			while rm_k >= np.ceil(self.case["task_type"][k][2]/RAM_UNIT) and rc_k > 0 and b_k[k] > be_k[k]:
				# set the w_ik to process +1
				wks[k] += 1
				# and take the resources on the available obs
				rm_k -= int(np.ceil(self.case["task_type"][k][2]/RAM_UNIT))
				rc_k -= 1
				be_k[k] += 1

		# offload to the Nearest Node if buffer bigger than 0.8
		for k,b in enumerate(b_k):
			if a_k[k] == 1:
				fks[k] = self.node.index
			if b_k[k] >= 0.8*MAX_QUEUE and a_k[k] == 1:
				# set the f_ik to the nearest node
				min_d = 10000; min_n = 0
				for n, d in self.node._distances.items():
					if d < min_d:
						min_n = n
						min_d = d
				fks[k] =min_n

		# and return the action
		return np.append(fks, wks)
