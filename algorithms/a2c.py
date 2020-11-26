#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame, Actor_Critic_Output_Frame
from algorithms.deep_tools.common import get_expected_returns
from algorithms.runners import run_episode, set_training_env

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE, DEFAULT_ACTION_SPACE
from sim_env.configs import N_NODES, DEFAULT_SLICES

# and external imports
import numpy as np
import tensorflow as tf
import tqdm
from typing import Tuple, List


# the class itself
class A2C_Agent(object):
	"""A2C_Agent
	"""
	basic = False
	def __init__(self, n,action_space=DEFAULT_ACTION_SPACE):
		super(A2C_Agent, self).__init__()
		# actual agent - the NN
		self.input_model = Simple_Frame()
		self.output_model = Actor_Critic_Output_Frame(action_space)
		
		# meta-data
		self.name = str(n)
		self.action_space = action_space
		self.learning_rate = DEFAULT_LEARNING_RATE
		self.gamma = 0.99


	def __call__(self, obs, batches=1):
		# wrapp in batches
		if batches == 1:
			obs = tf.expand_dims(obs, 0)
		# call its model
		action_logits_t,_ = self.output_model(self.input_model(obs))
		# and decipher the action
		action_i = []
		# Since it's multi-discrete, for every discrete set of actions:
		for action_logits_t_k in action_logits_t:
			# Sample next action from the action probability distribution
			action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
			action_i.append(action_i_k)
		# return the action for this agent
		return self.cap_to_action_space(action_i)

	def model(self, obs):
		return self.output_model(self.input_model(obs))

	def save_models(self, path):
		arch_path = path + str(self.input_model) + "/"
		self.input_model.save(arch_path+"input"+self.name)
		self.output_model.save(arch_path+"output"+self.name)
	def load_models(self, path):
		arch_path = path + str(self.input_model) + "/"
		self.input_model = tf.keras.models.load_model(arch_path+"input"+self.name, compile=False)
		self.output_model = tf.keras.models.load_model(arch_path+"output"+self.name, compile=False)
		
	def set_action_space(self, action_space):
		self.action_space = action_space
	def cap_to_action_space(self, action_i):
		for i in range(len(self.action_space)):
			if action_i[i] >= self.action_space[i]:
				action_i[i] = self.action_space[i]-1
		return action_i

	@staticmethod
	def short_str():
		return "a2c_"
	# to train RL agents  on an envrionment
	@staticmethod
	def train_agents_on_env(agents, env, max_episodes: int = 100, max_steps_per_episode: int = 1000):
		reward_threshold = 4500 # 0.9 ar, 5 nodes, 1000 timesteps (max)
		# Run the model for one episode to collect training data
		with tqdm.trange(max_episodes) as t:
			for i in t:
				initial_state = set_training_env(env)
				episode_reward = train_actor_critic(initial_state, agents, gamma=0.9, max_steps=max_steps_per_episode)

				t.set_description(f'Episode {i}')
				print(episode_reward)
				if episode_reward > reward_threshold:  
					break
		return agents


# optimizer to apply the gradient change
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# huber loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# combineed AC loss (with simple advantage)
def compute_combined_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
	advantages = returns-values
	action_log_probs = tf.math.log(action_probs)
	# T * T timesteps for 6 different actions - reduce sum to actor_loss[6]
	actor_loss = -tf.math.reduce_sum(action_log_probs * advantages, [0,1])
	critic_loss = huber_loss(values, returns)

	return actor_loss + critic_loss

# to train the advantage actor critic algorithm
@tf.function
def train_actor_critic(initial_state: tf.Tensor, agents: List[tf.keras.Model],
	optimizer: tf.keras.optimizers.Optimizer = optimizer, gamma: float = 0.99,
	max_steps: int = 10):

	with tf.GradientTape(persistent=True) as tape:
		
		action_losses = []; common_losses = []
		action_probs, values, rw = run_episode(initial_state, agents, max_steps)

		for i, agent in enumerate(agents):
			# compute stuff for each agent
			aux_val = tf.identity(values[i])
			aux_action_probs = tf.identity(action_probs[i])
			
			# Calculate simple advantage and returns
			returns = get_expected_returns(rw, aux_val, gamma)

			# Convert training data to appropriate TF tensor shapes
			aux_action_probs, aux_val, returns = [tf.expand_dims(x, 1) for x in [aux_action_probs, aux_val, returns]] 

			# Calculating loss values to update our network
			loss = compute_combined_loss(aux_action_probs, aux_val, returns)
			common_losses.append(tf.math.reduce_sum(loss))
			action_losses.append(loss)


	for agent, action_loss, common_loss in zip(agents, action_losses, common_losses):
		# Compute the gradients from the loss
		# for each output layer
		for layer in agent.output_model.output_layers:
			grads = tape.gradient(action_loss, layer.trainable_variables)
			optimizer.apply_gradients(zip(grads, layer.trainable_variables))
		# for the rest of the output layers
		for layer in agent.output_model.hidden_layers():
			grads = tape.gradient(common_loss, layer.trainable_variables)
			optimizer.apply_gradients(zip(grads, layer.trainable_variables))
		# and for the input layers
		grads = tape.gradient(common_loss, agent.input_model.trainable_variables)
		optimizer.apply_gradients(zip(grads, agent.input_model.trainable_variables))
	# drop reference to the tape
	del tape

	episode_reward = tf.math.reduce_sum(rw)
	return episode_reward
