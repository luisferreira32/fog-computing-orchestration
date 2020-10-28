# external openAI imports
import gym
from gym import spaces
# and necessary math libs
import numpy as np

# fog related imports



def CreatFogEnv(args):
	pass

class FogEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(FogEnv, self).__init__()
		# define the action space:
		# self.action_space = spaces.?

		# and the state space:
		# self.observation_space = spaces.?

		# self.seed()

		# self.nodes!


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