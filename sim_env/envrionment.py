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
from sim_env.configs import TIME_STEP, SIM_TIME_STEPS, RANDOM_SEED
from sim_env.configs import N_NODES, DEFAULT_SLICES, MAX_QUEUE, CPU_UNIT, RAM_UNIT
from sim_env.configs import PACKET_SIZE, BASE_SLICE_CHARS

# algorithm related imports
from algorithms.configs import OVERLOAD_WEIGHT

# for reproductibility
from utils.tools import set_seed


# ---------- Envrionment specific auxiliar functions ----------

# --- observation funs ---

def split_observation_by_logical_groups(obs):
	# splits the observations in the logical groups of the state: a_ik, b_ik, be_ik, rc_i, rm_i
	obs_by_nodes = split_observation_by_node(obs)
	a_ik = []; b_ik = []; be_ik = []; rc_i = []; rm_i = [];
	for i in range(N_NODES):
		[a, b, be, rc, rm] = np.split(obs_by_nodes[i], [DEFAULT_SLICES, DEFAULT_SLICES*2, DEFAULT_SLICES*3, DEFAULT_SLICES*3+1])
		a_ik = np.append(a_ik, a);
		b_ik = np.append(b_ik, b);
		be_ik = np.append(be_ik, be);
		rc_i.append(rc);
		rm_i.append(rm);
	return [a_ik, b_ik, be_ik, rc_i, rm_i]


def split_observation_by_node(obs):
	# splits the observation by nodes to several POMDP
	# [[a_00, ..., a_0k, b_00, ..., b_0k, be_00, ..., be_0k, rc_0, rm_0], ...
	# [a_i0, ..., a_ik, b_i0, ..., b_ik, be_i0, ..., be_ik, rc_i, rm_i]]
	return np.split(obs, N_NODES)

def split_node_obs_by_slice(nobs):
	# splits the node observation by slices to several POMDP
	pass


# --- action funs ---

def split_action_by_nodes(action):
	# splits the actions by nodes i
	# action space: [f_00, ..., f_0k, w_00, ..., w_0k, ..., f_i0, ..., f_ik, w_i0, ..., w_ik]
	return np.split(action, N_NODES)


# --- nodes funs ---

def get_nodes_characteristics(nodes):
	# returns the list of the total cpu units and ram units on nodes
	_cpu_units = [n.cpu_frequency/CPU_UNIT for n in nodes]
	_ram_units = [n.ram_size/RAM_UNIT for n in nodes]
	return [_cpu_units, _ram_units]

# ---------- Fog Envrionment ----------

class Fog_env(gym.Env):
	""" Fog_env looks to replicate a FC envrionment, configured in sim_env.configs
	it is a custom environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, case=BASE_SLICE_CHARS):
		super(Fog_env, self).__init__()

		# envrionment variables
		# self.nodes, self.evq, etc...
		self.nodes = [create_random_node(i, case) for i in range(N_NODES)]
		for n in self.nodes:
			n.set_communication_rates(self.nodes)
		self.evq = Event_queue()
		self.clock = 0
		self.saved_step_info = None

		# define the action space with I nodes and K slices each
		# [f_00, ..., f_0k, w_00, ..., w_0k, ..., f_i0, ..., f_ik, w_i0, ..., w_ik]
		# for each node there is an action [f_i0, ..., f_ik, w_i0, ..., w_ik]
		# where values can be between 0 and I for f_ik, and 0 and N for w_ik
		action_possibilities = []
		for n in self.nodes:
			for _ in range(n.max_k):
				action_possibilities.append(N_NODES) # f_ik: 0 to N_NODES-1 (nodes indexes)
			for _ in range(n.max_k):
				action_possibilities.append(n._avail_cpu_units+1) # w_ik: 0 to cpu_units
		action_possibilities = np.array(action_possibilities)
		self.action_space = spaces.MultiDiscrete(action_possibilities)

		# and the state space with I nodes and K slices each
		# [a_00, ..., a_0k, b_00, ..., b_0k, be_00, ..., be_0k, rc_0, rm_0,
		# ..., a_i0, ..., a_ik, b_i0, ..., b_ik, be_i0, ..., be_ik, rc_i, rm_i]
		# state_lows has to be remade if nodes don't have same slices
		state_possibilities = []
		for n in self.nodes:
			for _ in range(n.max_k):
				state_possibilities.append(2) # a_ik: 0 or 1
			for _ in range(n.max_k):
				state_possibilities.append(MAX_QUEUE+1) # b_ik: 0 to max_queue
			for _ in range(n.max_k):
				state_possibilities.append(n._avail_cpu_units+1) # be_ik: 0 to cpu_units
			state_possibilities.append(n._avail_cpu_units+1) # rc_i: 0 to cpu_units
			state_possibilities.append(n._avail_ram_units+1) # rm_i: 0 to ram_units
		state_possibilities = np.array(state_possibilities)
		self.observation_space = spaces.MultiDiscrete(state_possibilities)

		# Set up seeds for reproductibility
		self.seed(RANDOM_SEED)

		# and the first event that will trigger subsequent arrivals
		self.evq.addEvent(Set_arrivals(0, TIME_STEP, self.nodes))

	def seed(self, seed=None):
		# set all the necessary seeds for reproductibility
		self.np_random, seed = seeding.np_random(seed)
		self.action_space.seed(seed)
		self.observation_space.seed(seed)
		set_seed(seed)
		return [seed]

	def step(self, action):
		# to make sure you give actions in the FORMATED action space
		action = action.astype(np.uint8)
		assert self.action_space.contains(action)
		# current state
		state = self._next_observation()

		# information dict to pass back
		total = 0; overflow = 0; success = 0;
		info = {
			"discarded" : 0,
			"delay_list" : [],
			"success_rate": 0.0,
			"overflow_rate": 0.0,
			};

		# update some envrionment values
		for n in self.nodes:
			n.new_interval_update_service_rate()

		# calculate the instant rewards, based on state, action pair
		rw = self._reward_fun(state, action)

		# and execute the action
		self._take_action(action) 

		# increase the clock a timestep
		self.clock += TIME_STEP
		done = self.clock >= SIM_TIME_STEPS
		# rollout the events until new timestep
		while self.evq.hasEvents() and self.evq.first_time() <= self.clock:
			ev = self.evq.popEvent()
			t = ev.execute(self.evq)
			# update the info on offload sucess rate
			if ev.classtype == "Task_arrival" and ev.task_time() < self.clock:
				total += 1
				if t is None:
					success += 1
				elif not t.is_completed():
					overflow += 1
			# update info on completed (with delay) and discarded numbers
			if t is not None: # a task was returned
				if t.is_completed():
					info["delay_list"].append(t.task_delay())
				else:
					info["discarded"] += 1

		# save some info
		if total > 0:
			info["success_rate"] = (success)/(total)
			info["overflow_rate"] = (overflow)/(total)
		else:
			info["success_rate"] = []
			info["overflow_rate"] = []


		# obtain next observation
		obs = self._next_observation()

		# just save it for render
		self.saved_step_info = [obs, action]

		return obs, rw, done, info


	def reset(self):
		# Reset the state of the environment to an initial state
		self.evq.reset()
		self.evq.addEvent(Set_arrivals(0, TIME_STEP, self.nodes))
		for node in self.nodes:
			node.reset()	
		self.clock = 0
		self.seed(RANDOM_SEED)
		return self._next_observation()

	def _next_observation(self):
		# does a system observation
		obs = []
		for n in self.nodes:
			for k in range(n.max_k):
				if len(n.buffers[k]) > 0 and self.clock == n.buffers[k][-1]._timestamp:
				 	obs.append(1)
				else:
					obs.append(0)
			for k in range(n.max_k):
				obs.append(len(n.buffers[k]))
			for k in range(n.max_k):
				obs.append(n._being_processed[k])
			obs.append(n._avail_cpu_units)
			obs.append(n._avail_ram_units)
		return np.array(obs, dtype=np.uint8)

	def _take_action(self, action):
		# to make sure you give actions in the FORMATED action space
		action = action.astype(np.int8)
		# takes the action in the system, i.e. sets up the offloading events
		nodes_actions = split_action_by_nodes(action)
		for i in range(N_NODES):
			# for node i: [f_0, ..., f_k, w_0, ..., w_k]
			[fks, wks] = np.split(nodes_actions[i], 2)
			# concurrent offloads on this step
			con = len(fks)-np.count_nonzero(fks==i)
			for k in range(DEFAULT_SLICES):
				# if you are given a destination, add the offload event
				if fks[k] != i:
					self.evq.addEvent(Offload(self.clock, self.nodes[i], k, self.nodes[fks[k]], con))
				# and start processing if there is any request
				if wks[k] != 0:
					self.evq.addEvent(Start_processing(self.clock, self.nodes[i], k, wks[k]))

	def _reward_fun(self, state, action):
		# to make sure you give actions in the FORMATED action space
		action = action.astype(np.int8)
		# returns the instant reward of an action
		obs_by_nodes = split_observation_by_node(state)
		nodes_actions = split_action_by_nodes(action)

		# reward sum of all nodes:
		R = 0; 
		for obs, act, n in zip(obs_by_nodes, nodes_actions, self.nodes):
			node_reward = 0
			for k in range(n.max_k):
				D_ik = 0
				# if it's offloaded adds communication time to delay
				if act[k] != n.index:
					D_ik += PACKET_SIZE / n._communication_rates[act[k]]
				# calculate the Queue delay: b_ik/service_rate_i
				D_ik += obs[n.max_k+k]/n._service_rate
				# and the processing delay T*slice_k_cpu_demand / CPU_UNIT (GHz)
				D_ik +=  PACKET_SIZE*n._task_type_on_slices[k][1] / (CPU_UNIT*10**9)
				# finally, check if slice delay constraint is met
				if D_ik >= n._task_type_on_slices[k][0]:
					coeficient = -1
				else:
					coeficient = 1

				# also, verify if there is an overload chance in the arriving node
				if obs_by_nodes[act[k]][self.nodes[act[k]].max_k+k]+1 >= MAX_QUEUE:
					coeficient -= OVERLOAD_WEIGHT # tunable_weight

				# a_ik * ( (-1)if(delay_constraint_unmet) - (tunable_weight)if(overflow_chance) )
				node_reward += obs[k] * coeficient
			R += node_reward/n.max_k

		return R

	def render(self, mode='human', close=False):
		# Render the environment to the screen
		if self.saved_step_info is None: return
		nodes_obs = split_observation_by_node(self.saved_step_info[0])
		nodes_actions = split_action_by_nodes(self.saved_step_info[1])
		print("------",self.clock,"------")
		for i in range(N_NODES):
			[a, b, be, rc, rm] = np.split(nodes_obs[i], [DEFAULT_SLICES, DEFAULT_SLICES*2, DEFAULT_SLICES*3, DEFAULT_SLICES*3+1])
			print("n"+str(i)+" obs[ a:", a,"b:", b, "be:", be, "rc:",rc, "rm:",rm,"]")
			[f, w] = np.split(nodes_actions[i], 2)
			print("act[ f:",f,"w:",w,"]")
			for k,buf in enumerate(self.nodes[i].buffers):
				print("slice",k,"buffer",[round(t._timestamp,4) for t in buf])
		input("\nEnter to continue...")
		pass

