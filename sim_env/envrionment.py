#!/usr/bin/env python

# external openAI imports
import gym
from gym import spaces
from gym.utils import seeding

# and necessary math libs
import numpy as np

# fog related imports
from sim_env.core_classes import create_random_node
from sim_env.events import Event_queue, Set_arrivals, Offload, Start_processing
from sim_env.events_aux import is_arrival_on_slice
from sim_env.configs import TIME_STEP, SIM_TIME, RANDOM_SEED, OVERLOAD_WEIGHT
from sim_env.configs import N_NODES, DEFAULT_SLICES, MAX_QUEUE, CPU_UNIT, RAM_UNIT
from sim_env.configs import PACKET_SIZE, BASE_SLICE_CHARS

# for reproductibility
from utils.tools import set_tools_seed

# ---------- Fog Envrionment ----------

class Fog_env(gym.Env):
	""" Fog_env looks to replicate a FC envrionment, configured in sim_env.configs
	it is a custom environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, case=BASE_SLICE_CHARS, rd_seed=RANDOM_SEED):
		super(Fog_env, self).__init__()
		# Set up seeds for reproductibility
		self.seed(rd_seed)

		# envrionment variables
		# self.nodes, self.evq, etc...
		self.nodes = [create_random_node(i, case) for i in range(1,N_NODES+1)]
		for n in self.nodes:
			n.set_communication_rates(self.nodes)
		self.evq = Event_queue()
		self.clock = 0
		self.saved_step_info = None

		# define the action space with I nodes and K slices each
		# [[f_00, ..., f_0k, w_00, ..., w_0k], ..., [f_i0, ..., f_ik, w_i0, ..., w_ik]]
		# for each node there is an action [f_i0, ..., f_ik, w_i0, ..., w_ik]
		# where values can be between 0 and I for f_ik, and 0 and N=limited by either memory or cpu for w_ik
		action_possibilities = [np.append([N_NODES+1 for _ in range(n.max_k)],
			[min(n._avail_cpu_units, n._avail_ram_units)+1 for _ in range(n.max_k)]) for n in self.nodes]
		action_possibilities = np.array(action_possibilities, dtype=np.uint8)
		self.action_space = spaces.MultiDiscrete(action_possibilities)

		# and the state space with I nodes and K slices each
		# [a_00, ..., a_0k, b_00, ..., b_0k, be_00, ..., be_0k, rc_0, rm_0,
		# ..., a_i0, ..., a_ik, b_i0, ..., b_ik, be_i0, ..., be_ik, rc_i, rm_i]
		# state_lows has to be remade if nodes don't have same slices
		state_possibilities = [np.concatenate(([2 for _ in range(n.max_k)],[MAX_QUEUE+1 for _ in range(n.max_k)],
			[min(n._avail_cpu_units, n._avail_ram_units)+1 for _ in range(n.max_k)],
			[n._avail_cpu_units+1], [n._avail_ram_units+1])) for n in self.nodes]
		state_possibilities = np.array(state_possibilities, dtype=np.uint8)
		self.observation_space = spaces.MultiDiscrete(state_possibilities)

		# and the first event that will trigger subsequent arrivals
		self.evq.addEvent(Set_arrivals(0, TIME_STEP, self.nodes))

	def step(self, action_n):
		# to make sure you give actions in the FORMATED action space
		assert self.action_space.contains(action_n)
		state_t = self._get_state_obs()

		# to return it's necessary to return a lists in
		obs_n = [] # observations (POMDP)
		reward_n = [] # local rewards
		info_n = {} # information per agent

		# update some envrionment values
		for n in self.nodes:
			n.new_interval_update_service_rate()

		# measure instant reward of an action taken and queue it
		for i in range(N_NODES):
			# set the zeroed info
			info_n[i] = {
				"delay_list" : [],
				"overflow" : 0,
				"discarded" : 0,
				};

			# calculate the instant rewards, based on state, action pair
			rw = self._agent_reward_fun(self.nodes[i], self._get_agent_observation(n), action_n[i])
			reward_n.append(rw)

			# and execute the action
			self._set_agent_action(self.nodes[i], action_n[i]) 

		# increase the clock a timestep
		self.clock += TIME_STEP
		done = self.clock >= SIM_TIME
		# rollout the events until new timestep
		while self.evq.hasEvents() and self.evq.first_time() <= self.clock:
			ev = self.evq.popEvent()
			t = ev.execute(self.evq)

			# --- GET INFORMAITON HERE ---
			if t is not None: # means it came from a node
				if t.is_completed(): # finished
					info_n[n.index-1]["delay_list"].append(t.task_delay())
				elif t.task_time() == ev.time: # overflowed
					info_n[n.index-1]["overflow"] += 1
				else: # discarded because of delay constraint
					info_n[n.index-1]["discarded"] += 1


		# obtain next observation
		obs_n = self._get_state_obs()

		# just save it for render
		self.saved_step_info = [state_t, action_n]

		return obs_n, np.array(reward_n, dtype=np.float32), done, info_n


	def reset(self):
		# Reset the state of the environment to an initial state
		self.evq.reset()
		self.evq.addEvent(Set_arrivals(0, TIME_STEP, self.nodes))
		for node in self.nodes:
			node.reset()	
		self.clock = 0
		self.seed(RANDOM_SEED)
		return self._get_state_obs()

	def _get_state_obs(self):
		return np.array([self._get_agent_observation(n) for n in self.nodes], dtype=np.uint8)

	def _get_agent_observation(self, n):
		# does a partial system observation
		pobs = np.concatenate(([1 if len(n.buffers[k]) > 0 and self.clock == n.buffers[k][-1]._timestamp else 0 for k in range(n.max_k)],
			[len(n.buffers[k]) for k in range(n.max_k)], [n._being_processed[k] for k in range(n.max_k)],
			[n._avail_cpu_units],[n._avail_ram_units]))
		return np.array(pobs, dtype=np.uint8)

	def _set_agent_action(self, n, action):
		# takes the action in the system, i.e. sets up the offloading events
		# for node n: [f_0, ..., f_k, w_0, ..., w_k]
		[fks, wks] = np.split(action, 2)
		# concurrent offloads
		concurr = sum([1 if fk!=n.index and fk!=0 else 0 for fk in fks])
		for k in range(DEFAULT_SLICES):
			# start processing if there is any request
			if wks[k] != 0:
				self.evq.addEvent(Start_processing(self.clock, n, k, wks[k]))
			# and if you are given a destination, add the offload event
			if fks[k] != n.index and fks[k] != 0:
				self.evq.addEvent(Offload(self.clock, n, k, self.nodes[fks[k]-1], concurr))

	def _agent_reward_fun(self, n, obs, action):
		# calculate the reward for the agent (node) n
		node_reward = 0
		[fks, wks] = np.split(action, 2)
		concurr = sum([1 if fk!=n.index and fk!=0 else 0 for fk in fks])
		for k in range(n.max_k):
			D_ik = 0; Dt_ik = 0
			# if it's offloaded adds communication time to delay
			if fks[k] != n.index and fks[k] != 0:
				Dt_ik = PACKET_SIZE / (n._communication_rates[fks[k]-1]/concurr)
				D_ik += Dt_ik
			# calculate the Queue delay: b_ik/service_rate_i
			D_ik += obs[n.max_k+k]/n._service_rate
			# and the processing delay T*slice_k_cpu_demand / CPU_UNIT (GHz)
			D_ik +=  PACKET_SIZE*n._task_type_on_slices[k][1] / (CPU_UNIT*10**9)
			# finally, check if slice delay constraint is met
			if D_ik >= n._task_type_on_slices[k][0]:
				coeficient = -1
			else:
				coeficient = 1

			# count the number of new arrivals in the arriving node
			arr = 0
			for ev in self.evq.queue():
				if is_arrival_on_slice(ev, self.nodes[fks[k]-1], k) and ev.time <= self.clock+Dt_ik:
					arr += 1
			# also, verify if there is an overload chance in the arriving node
			arrival_node = self.nodes[fks[k]-1] if fks[k] > 0 else n
			arr_obs = self._get_agent_observation(arrival_node)
			if arr_obs[DEFAULT_SLICES+k]+arr+1 >= MAX_QUEUE:
				coeficient -= OVERLOAD_WEIGHT # tunable_weight

			# a_ik * ( (-1)if(delay_constraint_unmet) - (tunable_weight)if(overflow_chance) )
			node_reward += obs[k] * coeficient

		# float point reward
		return node_reward

	def render(self, mode='human', close=False):
		# Render the environment to the screen
		if self.saved_step_info is None: return
		nodes_obs = self.saved_step_info[0]
		nodes_actions = self.saved_step_info[1]
		print("------",round(self.clock*1000,2),"ms ------")
		for i in range(N_NODES):
			print("--- ",self.nodes[i]," ---")
			[a, b, be, rc, rm] = np.split(nodes_obs[i], [DEFAULT_SLICES, DEFAULT_SLICES*2, DEFAULT_SLICES*3, DEFAULT_SLICES*3+1])
			print(" obs[ a:", a,"b:", b, "be:", be, "rc:",rc, "rm:",rm,"]")
			[f, w] = np.split(nodes_actions[i], 2)
			print("act[ f:",f,"w:",w,"]")
			print("-- current state: --")
			for k,buf in enumerate(self.nodes[i].buffers):
				print("slice",k,"buffer",[round(t._timestamp,4) for t in buf])
		for ev in self.evq.queue():
			if ev.classtype != "Task_arrival" and ev.classtype != "Task_finished":
				print(ev.classtype+"["+str(round(ev.time*1000))+"ms]", end='-->')
		print(round(1000*(self.clock+TIME_STEP)),"ms")
		input("\nEnter to continue...")
		pass

	def seed(self, seed=None):
		# set all the necessary seeds for reproductibility
		# one for the sim_env random requirements
		self.np_random, seed = seeding.np_random(seed)
		# and another for the global randoms
		set_tools_seed(seed)
		return [seed]

