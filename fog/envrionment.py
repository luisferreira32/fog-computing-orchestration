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

	def __init__(self, nodes, sr=configs.SERVICE_RATE, ar=configs.TASK_ARRIVAL_RATE):
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
		self.ar = ar
		self.sr = sr
		self.evq = events.EventQueue()

		# and initial reset
		self.reset()



	def step(self, action):
		# information dict to pass back
		info = {}
		info["delays"] = []
		info["discarded"] = 0

		# calculate the instant rewards
		rw = self._reward_fun(action)
		# and execute the action, i.e., add the events
		self._take_action(action) 

		# Execute one time step within the environment
		self.clock += configs.TIME_INTERVAL
		done = self.clock >= configs.SIM_TIME

		# run all events until this time step
		while self.evq.hasEvents() and self.evq.first() < self.clock:
			ev = self.evq.popEvent()
			t = ev.execute(self.evq)
			if t is not None:
				if isinstance(t, int): info["discarded"] += t
				elif t.delay == -1: info["discarded"] += 1
				else: info["delays"].append(t.delay)
		
		# obtain next observation
		obs = self._next_observation()

		return obs, rw, done, info


	def reset(self):
		# Reset the state of the environment to an initial state
		self.clock = 0
		self.evq.reset()
		for n in self.nodes:
			n.reset()

		self.evq.addEvent(events.Recieving(0, self.nodes[0], ar=self.ar, 
			interval=configs.TIME_INTERVAL, nodes=self.nodes))
		
		return self._next_observation()

	def _next_observation(self):
		# does a system observation
		w_i = [len(n.w) for n in self.nodes]
		q_i = [n.qs() for n in self.nodes]
		npw_i = np.array(w_i, dtype=np.uint8)
		npq_i = np.array(q_i, dtype=np.uint8)
		obs = np.array((npw_i, npq_i)) # state

		return obs

	def _take_action(self, action):
		# takes the action in the system
		(n_io, w_io) = action

		for i, (n_o, w_o) in enumerate(zip(n_io, w_io)):
			nL = self.nodes[i]
			if not nL.transmitting:
				for x in range(w_o):
					nL.send(nL.decide(), self.nodes[n_o])
			while not nL.fullqueue() and nL.hasw():
				nL.queue(nL.decide())

		for n in self.nodes:
			# start processing if it hasn't started already
			if not n.processing and not n.emptyqueue():
				ev = events.Processing(self.clock, n)
				self.evq.addEvent(ev)
			# and sending if not sending already
			if not n.transmitting and n.tosend():
				ev = events.Sending(self.clock, n)
				self.evq.addEvent(ev)


	def _reward_fun(self, action):
		# returns the instant reward of an action
		(n_io, w_io) = action
		# given a state of the nodes
		w_i = [len(n.w) for n in self.nodes]
		q_i = [n.qs() for n in self.nodes]

		wLs = 0; wos = 0; t_c =0;
		for n_l, (w, q, n_o, w_o) in enumerate(zip(w_i, q_i, n_io, w_io)):
			wLs += min(configs.MAX_QUEUE - q, w - w_o)
			wos += w_o
			if w_o > 0:	t_c += w_o*self.nodes[n_l].comtime[self.nodes[n_o]]
		utility = 10 * np.log(1+wLs+wos)
		delay = 1 * t_c / (wLs + wos)

		return utility - delay


	def render(self, mode='human', close=False):
		# Render the environment to the screen
		pass

	"""def seed(self, seed=None):
	        self.np_random, seed = seeding.np_random(seed)
	        return [seed]"""