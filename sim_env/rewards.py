#!/usr/bin/env python
""" In this file reward formulas for the Fog_env openAI gym are formulated

"""

# >>>>> imports
import numpy as np

from sim_env.fog import point_to_point_transmission_rate
from sim_env.events import is_arrival_on_slice
from sim_env import configs as cfg

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> reward functions
def jbaek_reward_fun2(env, state, action, next_state, info):
	""" Calculates the reward for the system R(s,a,s')
	"""

	case = env.case
	rw = 0.0
	for s, a, n in zip(state, action, env.nodes):
		node_reward = 0.0
		[fks, wks] = np.split(a, 2)
		concurr = sum([1 if fk!=n.index and fk!=0 else 0 for fk in fks])

		for k in range(cfg.DEFAULT_SLICES):
			if s[k] == 1 and fks[k] != 0:
				# total delay and com delay
				D_ik = 0; Dt_ik = 0

				# if it's offloaded adds communication time to delay
				if fks[k] != n.index:
					bw = int(n.available_bandwidth()/concurr)
					if bw >= cfg.NODE_BANDWIDTH_UNIT: # just make sure it is actually going to offload
						Dt_ik = cfg.PACKET_SIZE / (point_to_point_transmission_rate(n._distances[fks[k]],bw))
						D_ik += Dt_ik*1000 # converto to milliseconds

				# calculate the Queue delay: b_ik/service_rate_i
				D_ik += s[n.max_k+k]/env.nodes[fks[k]-1]._service_rate[k] # service rate per millisecond
				# and the processing delay T*slice_k_cpu_demand / CPU_UNIT (GHz)
				D_ik +=  1000* cfg.PACKET_SIZE*case["task_type"][k][1] / (cfg.CPU_UNIT) # convert it to milliseconds
				# finally, check if slice delay constraint is met
				if D_ik >= case["task_type"][k][0]:
					coeficient = -1
				else:
					coeficient = 1

				# TODO@luis: solve this, how to accuratly predict if its going to be overlowed?
				# count the number of new arrivals in the arriving node 
				#arr = 0
				#for ev in env.evq.queue():
				#	if is_arrival_on_slice(ev, env.nodes[fks[k]-1], k) and ev.time <= env.clock+Dt_ik:
				#		arr += 1
				# also, verify if there is an overload chance in the arriving node
				#arr_obs = next_state[fks[k]-1]
				#if arr_obs[cfg.DEFAULT_SLICES+k]+arr+1 >= cfg.MAX_QUEUE:
				#	coeficient -= cfg.OVERLOAD_WEIGHT # tunable_weight

				# a_ik * ( (-1)if(delay_constraint_unmet) - (tunable_weight)if(overflow_chance) )
				node_reward += s[k] * coeficient

		# float point reward: 1/k * reward on each slice
		rw += node_reward/n.max_k
	return rw



def simple_reward(env, state, action, next_state, info):
	return len(info["delay_list"]) - info["overflow"]

def simple_reward2(env, state, action, next_state, info):
	return -info["overflow"]-info["discarded"]