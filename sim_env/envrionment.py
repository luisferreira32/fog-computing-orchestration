# external openAI imports
import gym
from gym import spaces
# and necessary math libs
import numpy as np

# fog related imports
from sim_env.core_classes import create_random_node
from sim_env.events import Event_queue
from sim_env.configs import N_NODES, DEFAULT_SLICES, MAX_QUEUE, CPU_UNIT, RAM_UNIT


def Create_fog_envrionment(args):
	pass

class Fog_env(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(Fog_env, self).__init__()
		# define the action space: [f_i,0, ..., f_i,k, w_i,0, ..., w_i,k]
		# self.action_space = spaces.?
		low_f = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		low_w = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		action_lows = np.append(low_f, low_w)

		high_f = np.full(DEFAULT_SLICES*N_NODES, N_NODES-1, dtype=np.uint8)
		high_w = np.full(DEFAULT_SLICES*N_NODES, MAX_QUEUE, dtype=np.uint8)
		action_highs = np.append(high_f, high_w)
		self.action_space = spaces.Box(low=action_lows, high=action_highs, dtype=np.uint8)

		# and the state space: [a_i,0, ..., a_i,k, b_i,0, ..., b_i,k, be_i,0, ..., be_i,k, r_i,c, r_i,m]
		# self.observation_space = spaces.?
		low_a = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		low_b = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		low_be = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
		state_lows = np.concatenate((low_a, low_b, low_be, [0], [0]), axis=0)

		high_a = np.full(DEFAULT_SLICES*N_NODES, 1, dtype=np.uint8)
		high_b = np.full(DEFAULT_SLICES*N_NODES, MAX_QUEUE, dtype=np.uint8)
		high_be = np.full(DEFAULT_SLICES*N_NODES, MAX_QUEUE, dtype=np.uint8)
		state_highs = np.concatenate((high_a, high_b, high_be, [10], [20]), axis=0)
		self.observation_space = spaces.Box(low=state_lows, high=state_highs, dtype=np.uint8)

		# self.seed()

		# envrionment variables
		# self.nodes, self.evq, etc...
		self.nodes = [create_random_node(i) for i in range(N_NODES)]
		self.evq = Event_queue()
		self.clock = 0


	def step(self, action):
		# information dict to pass back
		info = {};

		# calculate the instant rewards
		rw = self._reward_fun(action)

		# and execute the action
		self._take_action(action) 

		# rollout the events
		# update on nodes

		# obtain next observation
		obs = self._next_observation()

		return obs, rw, done, info


	def reset(self):
		# Reset the state of the environment to an initial state
		self.evq.reset()
		for node in self.nodes:
			node.reset()	
		self.clock = 0
		return self._next_observation()

	def _next_observation(self):
		# does a system observation
		a_ik = [1 if len(n.buffers[k]) > 0 and self.clock == n.buffers[k][-1]._timestamp else 0 for n in self.nodes for k in range(n.max_k)]
		b_ik = [len(n.buffers[k]) for n in self.nodes for k in range(n.max_k)]
		be_ik = [n._being_processed[k] for n in self.nodes for k in range(n.max_k)]
		r_ic = [np.uint8(n._avail_cpu_frequency/CPU_UNIT) for n in self.nodes]
		r_im = [np.uint8(n._avail_ram_size/RAM_UNIT) for n in self.nodes]
		obs = np.concatenate((a_ik, b_ik, be_ik, r_ic, r_im), axis=0)
		return obs

	def _take_action(self, action):
		# takes the action in the system
		pass

	def _reward_fun(self, action):
		# returns the instant reward of an action
		pass

	def render(self, mode='human', close=False):
		# Render the environment to the screen
		pass