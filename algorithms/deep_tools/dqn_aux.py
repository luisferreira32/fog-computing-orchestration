#!/usr/bin/env python

# >>>>> imports
import numpy as np

from sim_env.fog import point_to_point_transmission_rate
from sim_env import configs as cfg

def instant_rw_fun(env, state, action):
	""" Calculates the reward for the system R(s,a,s')
	"""

	case = env.case
	rw = 0.0
	estimated_trasmission_delays = []
	action = np.array(action)
	state = np.array(state)

	for s, a, n in zip(state, action, env.nodes):
		node_reward = 0.0
		[fks, wks] = np.split(a, 2)
		concurr = sum([1 if fk!=n.index and fk!=0 else 0 for fk in fks])

		node_estimated_trasmission_delays = []
		for k in range(cfg.DEFAULT_SLICES):
			if s[k] == 1 and fks[k] != 0:

				# If there is a transmission, get the transmission delay and the destination node itself
				Dt_ik = 0.0
				if fks[k] != n.index:
					bw = int(n.available_bandwidth()/concurr)
					if bw >= cfg.NODE_BANDWIDTH_UNIT: # just make sure it is actually going to offload
						Dt_ik = cfg.PACKET_SIZE / (point_to_point_transmission_rate(n._distances[fks[k]],bw)) # in [s]
				node_estimated_trasmission_delays.append(Dt_ik)

				# Calculate the total processing delay in [ms]
				# which is the transmission delay (=0.0 if no offload)
				D_ik = Dt_ik*1000
				# calculate the Queue delay: b_ik/service_rate_i
				D_ik += s[cfg.DEFAULT_SLICES+k]/n._service_rate[k] # service rate is per millisecond
				# and the processing delay T*slice_k_cpu_demand / CPU_UNIT (GHz)
				D_ik +=  1000* cfg.PACKET_SIZE*case["task_type"][k][1] / (cfg.CPU_UNIT) # convert it to milliseconds
				# finally, check if slice delay constraint is met
				if D_ik >= case["task_type"][k][0]:
					coeficient = -1
				else:
					coeficient = 1

				# a_ik * ( (-1)if(delay_constraint_unmet) - (tunable_weight)if(overflow_chance) )
				node_reward += s[k] * coeficient

		# float point reward: 1/k * reward on each slice
		rw += node_reward/n.max_k
		estimated_trasmission_delays.append(node_estimated_trasmission_delays)
	return rw, estimated_trasmission_delays


# for a limited size replay buffer
class Replay_buffer(object):
	"""docstring for Replay_buffer"""
	def __init__(self, max_size=-1):
		super(Replay_buffer, self).__init__()
		self.max_size = max_size
		self.state_buffer = []
		self.action_buffer = []
		self.rw_buffer = []
		self.next_state_buffer = []

	def push(self, state, action, rw, next_state):
		if len(self.state_buffer) == self.max_size:
			self.state_buffer.pop(0)
		if len(self.action_buffer) == self.max_size:
			self.action_buffer.pop(0)
		if len(self.rw_buffer) == self.max_size:
			self.rw_buffer.pop(0)
		if len(self.next_state_buffer) == self.max_size:
			self.next_state_buffer.pop(0)

		self.state_buffer.append(state)
		self.action_buffer.append(action)
		self.rw_buffer.append(rw)
		self.next_state_buffer.append(next_state)

	def get_tuple(self):
		return (self.state_buffer, self.action_buffer, self.rw_buffer, self.next_state_buffer)

	def size(self):
		return len(self.state_buffer)


class Temporary_experience(object):
	"""docstring for Temporary_experience"""
	def __init__(self, state, action, instant_rw, next_state, init_time, transmission_delays):
		super(Temporary_experience, self).__init__()
		self.state = state
		self.action = action
		self.instant_rw = instant_rw
		self.next_state = next_state

		self.times = []
		for node, node_times in enumerate(transmission_delays):
			for slice_k, slice_node_times in enumerate(node_times):
				if slice_node_times != 0.0:
					self.times.append([node, slice_k, slice_node_times+init_time])

	def value_tuple(self):
		return (self.state, self.action, self.instant_rw, self.next_state)

	def check_update(self, current_time, obs_n):
		finished = False
		for elem in self.times:
			n, k, t = elem
			if t <= current_time:
				self.times.remove(elem)
				if obs_n[n][cfg.DEFAULT_SLICES+k]+1 >= cfg.MAX_QUEUE:
					self.instant_rw -= cfg.OVERLOAD_WEIGHT/cfg.DEFAULT_SLICES
		if len(self.times) == 0:
			finished = True

		return finished

