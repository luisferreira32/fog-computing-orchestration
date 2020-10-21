# external openAI imports
import gym
from gym import spaces
# and necessary math libs
import numpy as np

# fog related imports
from . import configs
from . import events

# constants

class FogEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, nodes, ar=configs.TASK_ARRIVAL_RATE, sr=configs.SERVICE_RATE):
		super(FogEnv, self).__init__()
		# define the action space: all possible (no, wo) combinations
		self.action_space = spaces.Box(low=0, high=np.array([configs.N_NODES, configs.MAX_W]), dtype=np.uint8)

		# and the state space: (nL, w, Q_1, Q_2, ..., Q_N )
		max_Qs = np.full(configs.N_NODES, configs.MAX_QUEUE, dtype=np.uint8)
		high = np.array([configs.N_NODES, configs.MAX_W])
		high = np.append(high, max_Qs)
		self.observation_space = spaces.Box(low=0, high=high, dtype=np.uint8)

		# env related variables
		self.clock = 0
		self.nodes = nodes
		self.evq = events.EventQueue()
		evq.addEvent(events.Recieving(0, self.nodes[0], ar=ar, interval=configs.TIME_INTERVAL, nodes=self.nodes))


	def step(self, action):
		# Execute one time step within the environment
		self.clock += configs.TIME_INTERVAL
		# run all events until this timestep
		while evq.hasEvents() and evq.first() < self.clock:
			ev = evq.popEvent()
			ev.execute(evq)
		# and execute the action, i.e., add the events

		pass

	def reset(self):
		# Reset the state of the environment to an initial state
		self.clock = 0
		self.evq.reset()
		for n in self.nodes:
			n.reset()

		evq.addEvent(events.Recieving(0, self.nodes[0], ar=ar, interval=configs.TIME_INTERVAL, nodes=self.nodes))
		
		return self._next_observation()

	def _next_observation(self):
		# does a system observation
		pass

	def _take_action(self):
		# takes the action in the system
		pass

	def render(self, mode='human', close=False):
		# Render the environment to the screen
		pass
