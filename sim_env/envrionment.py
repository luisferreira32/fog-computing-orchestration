#!/usr/bin/env python

# external openAI imports
import gym
from gym import spaces
# and necessary math libs
import numpy as np

# fog related imports
from sim_env.core_classes import create_random_node
from sim_env.events import Event_queue, Set_arrivals, Offload, Start_processing
from sim_env.configs import TIME_STEP, SIM_TIME_STEPS
from sim_env.configs import N_NODES, DEFAULT_SLICES, MAX_QUEUE, CPU_UNIT, RAM_UNIT


def Create_fog_envrionment(args):
	pass

class Fog_env(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(Fog_env, self).__init__()

		# envrionment variables
		# self.nodes, self.evq, etc...
		self.nodes = [create_random_node(i) for i in range(N_NODES)]
		for n in self.nodes:
			n.set_communication_rates(self.nodes)
		self.evq = Event_queue()
		self.clock = 0

		# define the action space with I nodes and K slices each
		# [f_00, ..., f_0k, w_00, ..., w_0k, ..., f_i0, ..., f_ik, w_i0, ..., w_ik]
		# action_lows has to be remade if nodes don't have same slices
		action_lows = np.zeros(DEFAULT_SLICES*N_NODES*2, dtype=np.uint8)

		action_highs = []
		for n in self.nodes:
			for _ in range(n.max_k):
				action_highs.append(np.uint8(N_NODES-1))
			for _ in range(n.max_k):
				action_highs.append(np.uint8(n._avail_cpu_units))
		action_highs = np.array(action_highs)
		self.action_space = spaces.Box(low=action_lows, high=action_highs, dtype=np.uint8)

		# and the state space with I nodes and K slices each
		# [a_00, ..., a_0k, b_00, ..., b_0k, be_00, ..., be_0k, rc_0, rm_0,
		# ..., a_i0, ..., a_ik, b_i0, ..., b_ik, be_i0, ..., be_ik, rc_i, rm_i]
		# state_lows has to be remade if nodes don't have same slices
		state_lows = np.zeros(DEFAULT_SLICES*N_NODES*3+N_NODES*2, dtype=np.uint8)

		state_highs = []
		for n in self.nodes:
			for _ in range(n.max_k):
				state_highs.append(1) # a_ik
			for _ in range(n.max_k):
				state_highs.append(MAX_QUEUE) # b_ik
			for _ in range(n.max_k):
				state_highs.append(np.uint8(n._avail_cpu_units)) # be_ik
			state_highs.append(np.uint8(n._avail_cpu_units)) # rc_i
			state_highs.append(np.uint8(n._avail_ram_units)) # rm_i
		state_highs = np.array(state_highs)
		self.observation_space = spaces.Box(low=state_lows, high=state_highs, dtype=np.uint8)

		# self.seed()

		# and the first event that will trigger subsequent arrivals
		self.evq.addEvent(Set_arrivals(0, TIME_STEP, self.nodes))


	def step(self, action):
		# information dict to pass back
		info = {
			"discarded" : 0,
			"delay_list" : []
			};

		# calculate the instant rewards, based on state, action pair
		rw = self._reward_fun(self._next_observation(), action)

		# and execute the action
		self._take_action(action) 

		# increase the clock a timestep
		self.clock += TIME_STEP
		done = self.clock >= SIM_TIME_STEPS
		# rollout the events until new timestep
		while self.evq.hasEvents() and self.evq.first_time() < self.clock:
			ev = self.evq.popEvent()
			t = ev.execute(self.evq)
			# update the info based on the object task returning
			if t is not None: # a task was returned
				if t.is_completed():
					info["delay_list"].append(t.task_delay())
				else:
					info["discarded"] += 1
		# which updates states on the nodes

		# obtain next observation
		obs = self._next_observation()

		return obs, rw, done, info


	def reset(self):
		# Reset the state of the environment to an initial state
		self.evq.reset()
		self.evq.addEvent(Set_arrivals(0, TIME_STEP, self.nodes))
		for node in self.nodes:
			node.reset()	
		self.clock = 0
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
			for k in range(DEFAULT_SLICES):
				# if you are given a destination, add the offload event
				if fks[k] != i:
					self.evq.addEvent(Offload(self.clock, self.nodes[i], k, self.nodes[fks[k]]))
				# and start processing if there is any request
				if wks[k] != 0:
					self.evq.addEvent(Start_processing(self.clock, self.nodes[i], k, wks[k]))

	def _reward_fun(self, state, action):
		# returns the instant reward of an action
		return 0

	def render(self, mode='human', close=False):
		# Render the environment to the screen
		"""
		nodes_obs = split_observation_by_node(self._next_observation())
		print("------",self.clock,"------")
		for i, obs in enumerate(nodes_obs):
			print("n"+str(i)+" obs:", obs)
		"""
		pass

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
	return np.split(obs, N_NODES)

def split_observation_by_slices(obs):
	# splits the observation by slices to several POMDP
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