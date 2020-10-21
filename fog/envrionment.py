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
		# define the action space: all possible (n_1o, ..., n_No, w_1o, ..., w_No) combinations
		max_Ns = np.full(configs.N_NODES, configs.N_NODES, dtype=np.uint8)
		max_Ws = np.full(configs.N_NODES, configs.MAX_W, dtype=np.uint8)
		self.action_space = spaces.Tuple((
			spaces.Box(low=0, high=max_Ns, dtype=np.uint8),
			spaces.Box(low=0, high=max_Ws, dtype=np.uint8)))

		# and the state space: (w_1, ..., w_N, Q_1, Q_2, ..., Q_N)
		max_Qs = np.full(configs.N_NODES, configs.MAX_QUEUE, dtype=np.uint8)
		max_Ws = np.full(configs.N_NODES, configs.MAX_W, dtype=np.uint8)
		self.observation_space = spaces.Tuple((
			spaces.Box(low=0, high=max_Ws, dtype=np.uint8),
			spaces.Box(low=0, high=max_Qs, dtype=np.uint8)))

		self.seed()

		# env related variables
		self.clock = 0
		self.nodes = nodes
		self.evq = events.EventQueue()

		# and initial reset
		self.reset()



	def step(self, action):
		# Execute one time step within the environment
		self.clock += configs.TIME_INTERVAL
		done = self.clock >= configs.SIM_TIME


		# calculate the instant rewards
		rw = self._reward_fun(action)
		# and execute the action, i.e., add the events
		self._take_action(action) 

		# run all events until this time step
		while self.evq.hasEvents() and self.evq.first() < self.clock:
			ev = self.evq.popEvent()
			ev.execute(self.evq)
		
		# obtain next observation
		obs = self._next_observation()

		return obs, rw, done, {}


	def reset(self):
		# Reset the state of the environment to an initial state
		self.clock = 0
		self.evq.reset()
		for n in self.nodes:
			n.reset()

		evq.addEvent(events.Recieving(0, self.nodes[0], ar=ar, 
			interval=configs.TIME_INTERVAL, nodes=self.nodes))
		
		return self._next_observation()

	def _next_observation(self):
		# does a system observation
		pass

	def _take_action(self):
		# takes the action in the system
		pass

	def _reward_fun(self, action):
		# returns the instant reward of an action
		(n_io, w_io) = action
		# given a state of the nodes
		w_i = [len(n.w) for n in self.nodes]
		q_i = [n.qs() for n in self.nodes]

		for w, q, w_o in zip(w_i, q_i, n_io):
			wL = max(configs.MAX_QUEUE - q, w - w_o)
			utility += 10 * np.log()


	def render(self, mode='human', close=False):
		# Render the environment to the screen
		pass

	def seed(self, seed=None):
	        self.np_random, seed = seeding.np_random(seed)
	        return [seed]