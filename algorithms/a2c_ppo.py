#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import set_tf_seed, Simple_Frame, Actor_Critic_Output_Frame

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE, DEFAULT_PPO_EPS
from sim_env.configs import N_NODES

# and mathematical help
import numpy as np
import tensorflow as tf

# and to make it reproducible
set_tf_seed(ALGORITHM_SEED)

# huber loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
# and adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# the class itself
class A2C_PPO_Agent(object):
	"""A2C_PPO_Agent
	"""
	def __init__(self, n, ppo_eps=DEFAULT_PPO_EPS):
		super(A2C_PPO_Agent, self).__init__()
		action_possibilities = np.append([N_NODES+1 for _ in range(n.max_k)],
			[min(n._avail_cpu_units, n._avail_ram_units)+1 for _ in range(n.max_k)])
		action_possibilities = np.array(action_possibilities, dtype=np.uint8)
		# actual agent - the NN
		self.model = Simple_Frame(action_possibilities, output_frame=Actor_Critic_Output_Frame)
		
		# meta-data
		self.learning_rate = DEFAULT_LEARNING_RATE
		self.ppo_eps = ppo_eps


	def __call__(self, obs, batches=1):
		# wrapp in batches
		if batches == 1:
			obs = tf.expand_dims(obs, 0)
		# call its model
		action_logits_t,_ = self.model(obs)
		# and decipher the action
		action_i = []
		# Since it's multi-discrete, for every discrete set of actions:
		for action_logits_t_k in action_logits_t:
			# Sample next action from the action probability distribution
			action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
			action_i.append(action_i_k.numpy())
		# return the action for this agent
		return np.array(action_i)

	def compute_combined_loss( action_probs: tf.Tensor, advantages: tf.Tensor, 
		returns: tf.Tensor) -> tf.Tensor:
		action_log_probs = tf.math.log(action_probs)
		actor_loss = 0 #self._ppo_loss()
		critic_loss = huber_loss(values, returns)

		return actor_loss + critic_loss

	def _ppo_loss(self):
		pass
