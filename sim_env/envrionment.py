# external openAI imports
import gym
from gym import spaces
# and necessary math libs
import numpy as np

# fog related imports
from sim_env.core_classes import create_random_node
from sim_env.events import Event_queue
from sim_env.configs import N_NODES, DEFAULT_SLICES, MAX_QUEUE


def CreatFogEnv(args):
	pass

class FogEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(FogEnv, self).__init__()
		# define the action space: [f_0, ..., f_k, w_0, ..., w_k]
		# self.action_space = spaces.?
		low_f = np.zeros(DEFAULT_SLICES, dtype=np.uint8)
		low_w = np.zeros(DEFAULT_SLICES, dtype=np.uint8)
		action_lows = np.append(low_f, low_w)

		high_f = np.full(DEFAULT_SLICES, N_NODES-1, dtype=np.uint8)
		high_w = np.full(DEFAULT_SLICES, MAX_QUEUE, dtype=np.uint8)
		action_highs = np.append(high_f, high_w)
		self.action_space = space.Box(low=action_lows, high=action_highs, dtype=np.uint8)

		# and the state space: [a_0, ..., a_k, b_0, ..., b_k, be_0, ..., be_k, r_c, r_m]
		# self.observation_space = spaces.?
		low_a = np.zeros(DEFAULT_SLICES, dtype=np.uint8)
		low_b = np.zeros(DEFAULT_SLICES, dtype=np.uint8)
		low_be = np.zeros(DEFAULT_SLICES, dtype=np.uint8)
		state_lows = np.append(low_a, low_b, low_be, 0, 0)

		high_a = np.full(DEFAULT_SLICES, 1, dtype=np.uint8)
		high_b = np.full(DEFAULT_SLICES, MAX_QUEUE, dtype=np.uint8)
		high_be = np.full(DEFAULT_SLICES, MAX_QUEUE, dtype=np.uint8)
		state_highs = np.append(high_a, high_b, high_be, 10, 20)
		self.observation_space = spaces.Box(low=state_lows, high=state_highs, dtype=np.uint8)

		# self.seed()

		# envrionment variables
		# self.nodes, self.evq, etc...
		self.nodes = [create_random_node(i) for i in range(N_NODES)]
		self.evq = Event_queue()


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
		return self._next_observation()

	def _next_observation(self):
		# does a system observation
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