# external openAI imports
import gym
from gym import spaces
# and necessary math libs
import numpy as np

# fog related imports
from fog import configs, events, node
from tools import utils

def CreatFogEnv(sr=configs.SERVICE_RATE, ar=configs.TASK_ARRIVAL_RATE):
	utils.initRandom()
	# placement of the nodes
	placements=[]
	for i in range(0, configs.N_NODES):
		placements.append((utils.uniformRandom(configs.MAX_AREA[0]),utils.uniformRandom(configs.MAX_AREA[1])))

	# the nodes 
	cps = sr*configs.DEFAULT_IL*configs.DEFAULT_CPI/configs.TIME_INTERVAL
	nodes = []
	for i in range(0, configs.N_NODES):
		n = node.Core(name="n"+str(i), index=i,	placement=placements[i], cpu=(configs.DEFAULT_CPI, cps))
		nodes.append(n)
	# create M edges between each two nodes
	for n in nodes:
		n.setcomtime(nodes)

	return FogEnv(nodes, sr, ar)

class FogEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, nodes, sr=configs.SERVICE_RATE, ar=configs.TASK_ARRIVAL_RATE):
		super(FogEnv, self).__init__()
		# define the action space: all possible (n_1o, ..., n_No, w_1o, ..., w_No) combinations
		max_Ns = np.full(configs.N_NODES, configs.N_NODES-1, dtype=np.uint8)
		max_Ws = np.full(configs.N_NODES, configs.MAX_W, dtype=np.uint8)
		high = np.append(max_Ns, max_Ws)
		self.action_space = spaces.Box(low=0, high=high, dtype=np.uint8)

		# and the state space: (w_1, ..., w_N, Q_1, Q_2, ..., Q_N)
		max_Ws = np.full(configs.N_NODES, configs.MAX_W, dtype=np.uint8)
		max_Qs = np.full(configs.N_NODES, configs.MAX_QUEUE, dtype=np.uint8)
		high = np.append(max_Ws, max_Qs)
		self.observation_space = spaces.Box(low=0, high=high, dtype=np.uint8)

		self.seed()

		# env related variables
		self.clock = 0
		self.nodes = nodes
		self.ar = ar
		self.sr = sr
		self.evq = events.EventQueue()


	def step(self, action):
		# information dict to pass back
		info = {}; info["delays"] = []; info["discarded"] = 0

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
		obs = np.append(npw_i, npq_i) # state

		return obs

	def _take_action(self, action):
		# takes the action in the system
		action = action.astype(np.uint8) # brute force fix
		(n_io, w_io) = np.split(action, 2)

		for i, (n_o, w_o) in enumerate(zip(n_io, w_io)):
			nL = self.nodes[i]
			if not nL.transmitting and i != n_o:
				for x in range(w_o):
					if not nL.hasw(): break
					nL.send(nL.decide(), self.nodes[n_o])
			while not nL.fullqueue() and nL.hasw():
				nL.queue(nL.decide())

		for n in self.nodes:
			if n.hasw():
				discarded = len(n.w)
				n.w.clear()
				self.evq.addEvent(events.Discard(self.clock, discarded))
			# start processing if it hasn't started already
			if not n.processing and not n.emptyqueue():
				self.evq.addEvent(events.Processing(self.clock, n))
			# and sending if not sending already
			if not n.transmitting and n.tosend():
				self.evq.addEvent(events.Sending(self.clock, n))


	def _reward_fun(self, action):
		# returns the instant reward of an action
		action = action.astype(np.uint8) # brute force fix
		(n_io, w_io) = np.split(action,2)
		# given a state of the nodes
		w_i = [len(n.w) for n in self.nodes]
		q_i = [n.qs() for n in self.nodes]

		wLs = 0; wos = 0; t_c =0;
		for n_l, (w, q, n_o, w_o) in enumerate(zip(w_i, q_i, n_io, w_io)):
			# if there is an impossible action, penalize it infinitely
			if w_o > w: return -1000.0
			# else calculate normal reward
			wLs += min(configs.MAX_QUEUE - q, w - w_o)
			if w_o > 0 and n_l != n_o: t_c += w_o*self.nodes[n_l].comtime[self.nodes[n_o]]
		wos = np.sum(w_io)
		utility = 10 * np.log(1+wLs+wos)
		delay = 1 * t_c / (wLs + wos + 1)

		return utility - delay


	def render(self, mode='human', close=False):
		# Render the environment to the screen
		pass

	"""def seed(self, seed=None):
	        self.np_random, seed = seeding.np_random(seed)
	        return [seed]"""