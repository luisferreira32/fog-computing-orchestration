# external openAI imports
import gym
from gym import spaces
# and necessary math libs
import numpy as np

# fog related imports
from . import configs
from . import events

# constants
N_DISCRETE_ACTIONS = configs.N_NODES*configs.MAX_W

class FogEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, nodes):
		super(FogEnv, self).__init__()
		# define the action space: all possible (no, wo) combinations
		self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
		# and the state space: (nL, w, Q_1, Q_2, ..., Q_N )
		max_Qs = np.full(configs.N_NODES, configs.MAX_QUEUE, dtype=int)
		high = np.array([configs.N_NODES, configs.MAX_W])
		high = np.append(high, max_Qs)
		self.observation_space = spaces.Box(low=0, high=high, dtype=np.uint8)

		# env related variables
		self.clock = 0
		self.nodes = nodes
		self.evq = events.EventQueue()


	def step(self, action):
		# Execute one time step within the environment
		pass

	def reset(self):
		# Reset the state of the environment to an initial state
		self.clock = 0
		self.evq.reset()
		for n in self.nodes:
			n.reset()
		pass

	def render(self, mode='human', close=False):
		# Render the environment to the screen
		pass
