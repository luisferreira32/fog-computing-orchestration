from . import configs
from algorithms import basic, qlearning

class Controller(object):
	"""docstring for Controller"""
	def __init__(self, nodes, algorithm_object):
		self.nodes = nodes
		self.algorithm_object = algorithm_object
		self.past = {}

	def assert_state(self, nL):
		Qs = []
		for n in self.nodes: Qs.append(n.qs())
		return [nL, len(nL.w), Qs]

	def decide(self):
		discarded = 0
		# each timestep
		if self.algorithm_object.updatable:
			self.algorithm_object.changeiter(epsilon=self.algorithm_object.epsilon-0.85/(configs.SIM_TIME/configs.TIME_INTERVAL))

		for nL in self.nodes:
			# decide for current node with incoming tasks
			if len(nL.w) > 0:
				# state = (nL, w, Qsizes)
				state = self.assert_state(nL)
				instant_reward = 0
				# if it is updatable do it before taking a new action with the trio (s, a, s')
				if self.algorithm_object.updatable and nL in self.past:
					self.algorithm_object.update(self.past[nL][0], self.past[nL][1], state, 
						self.past[nL][2], self.nodes)

				# run the algorithm
				(w0, nO) = self.algorithm_object.execute(state)
				# and save past state if there was an action taken
				if self.algorithm_object.updatable and nO is not None:
					instant_reward = self.algorithm_object.reward(state, [w0, nO])
					self.past[nL] = [state, [w0, nO], instant_reward]

				# and execute the decision, but since there's no sending queue, only when not transmitting
				if not nL.transmitting:
					for i in range(w0):
						nL.send(nL.decide(), nO)
				for j in range(len(nL.w)):
					if not nL.fullqueue(): nL.queue(nL.decide())

			# what to do with the rest of the w?
			discarded += len(nL.w)
			nL.w.clear()

		return discarded
