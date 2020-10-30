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
		# [f_00, f_01, ..., f_ik, w_00, w_01, ..., w_ik]
		# self.action_space = spaces.?
		low_f = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		low_w = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		action_lows = np.append(low_f, low_w)

		high_f = np.full(DEFAULT_SLICES*N_NODES, N_NODES-1, dtype=np.uint8)
		high_w = [ np.uint8(n._avail_cpu_units) for n in self.nodes for k in range(n.max_k)]
		action_highs = np.append(high_f, high_w)
		self.action_space = spaces.Box(low=action_lows, high=action_highs, dtype=np.uint8)

		# and the state space with I nodes and K slices each
		# [a_00, a_01, ..., a_ik, b_00, b_01 ..., b_ik, be_00, be_01, ..., be_ik, rc_0, ..., rc_i, rm_0, ..., rm_i]
		# self.observation_space = spaces.?
		low_a = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		low_b = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		low_be = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		low_rc = np.zeros(N_NODES, dtype=np.uint8)
		low_rm = np.zeros(N_NODES, dtype=np.uint8)
		state_lows = np.concatenate((low_a, low_b, low_be, low_rc, low_rm), axis=0)

		high_a = np.full(DEFAULT_SLICES*N_NODES, 1, dtype=np.uint8)
		high_b = np.full(DEFAULT_SLICES*N_NODES, MAX_QUEUE, dtype=np.uint8)
		high_be = [ np.uint8(n._avail_cpu_units) for n in self.nodes for k in range(n.max_k)]
		high_rc = [ np.uint8(n._avail_cpu_units) for n in self.nodes]
		high_rm = [ np.uint8(n._avail_ram_units) for n in self.nodes]
		state_highs = np.concatenate((high_a, high_b, high_be, high_rc, high_rm), axis=0)
		self.observation_space = spaces.Box(low=state_lows, high=state_highs, dtype=np.uint8)

		# self.seed()

		# and the first event that will trigger subsequent arrivals
		self.evq.addEvent(Set_arrivals(0, TIME_STEP, self.nodes))


	def step(self, action):
		# information dict to pass back
		info = {};

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
		a_ik = [1 if len(n.buffers[k]) > 0 and self.clock == n.buffers[k][-1]._timestamp else 0 for n in self.nodes for k in range(n.max_k)]
		b_ik = [len(n.buffers[k]) for n in self.nodes for k in range(n.max_k)]
		be_ik = [n._being_processed[k] for n in self.nodes for k in range(n.max_k)]
		rc_i = [n._avail_cpu_units for n in self.nodes]
		rm_i = [n._avail_ram_units for n in self.nodes]
		obs = np.concatenate((a_ik, b_ik, be_ik, rc_i, rm_i), axis=0)
		return obs

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
	return np.split(obs, [DEFAULT_SLICES*N_NODES, 2*DEFAULT_SLICES*N_NODES, 3*DEFAULT_SLICES*N_NODES,
		3*DEFAULT_SLICES*N_NODES+N_NODES])

def split_observation_by_node(obs):
	# splits the observation by nodes to several POMDP
	[a_ik, b_ik, be_ik, r_ic, r_im] = split_observation_by_logical_groups(obs)
	aks = np.split(a_ik, N_NODES)
	bks = np.split(b_ik, N_NODES)
	beks = np.split(be_ik, N_NODES)
	rc = np.split(r_ic, N_NODES)
	rm = np.split(r_im, N_NODES)
	return [ np.concatenate((aks[i], bks[i], beks[i], rc[i], rm[i])) for i in range(N_NODES) ]

def split_observation_by_slices(obs):
	# splits the observation by slices to several POMDP
	pass


# --- action funs ---

def split_action_by_nodes(action):
	# splits the actions by nodes i
	# action space: [f_00, f_01, ..., f_ik, w_00, w_01, ..., w_ik]
	[fs, ws] = np.split(action, 2)
	fsi = np.split(fs, N_NODES)
	wsi = np.split(ws, N_NODES)
	return [ np.append(fsi[i],wsi[i]) for i in range(N_NODES)]


# --- nodes funs ---

def get_nodes_characteristics(nodes):
	# returns the list of the total cpu units and ram units on nodes
	_cpu_units = [n.cpu_frequency/CPU_UNIT for n in nodes]
	_ram_units = [n.ram_size/RAM_UNIT for n in nodes]
	return [_cpu_units, _ram_units]