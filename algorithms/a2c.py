#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame
from algorithms.deep_tools.common import get_expected_returns
from algorithms.trainners import run_tragectory, set_training_env

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE, DEFAULT_ACTION_SPACE
from sim_env.configs import N_NODES, DEFAULT_SLICES, TOTAL_TIME_STEPS

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
	def __init__(self, n, action_space=DEFAULT_ACTION_SPACE, model_frame=Simple_Frame):
		super(A2C_Agent, self).__init__()
		# actual agent - the NN
		self.model = model_frame(action_space)		
		# meta-data
		self.name = "node_"+str(n)+"_agent"
		self.action_space = action_space
		self.learning_rate = DEFAULT_LEARNING_RATE
		self.gamma = 0.99

	def __str__(self):
		return self.name


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
			action_i.append(action_i_k)
		# return the action for this agent
		return action_i

	def model(self, obs):
		return self.model(obs)

	def save_models(self, path):
		arch_path = path + str(self.input_model) + "/"
		self.input_model.save(arch_path+self.name+"_input")
		self.output_model.save(arch_path+self.name+"_output")
	def load_models(self, path):
		arch_path = path + str(self.input_model) + "/"
		self.input_model = tf.keras.models.load_model(arch_path+self.name+"_input", compile=False)
		self.output_model = tf.keras.models.load_model(arch_path+self.name+"_output", compile=False)
		

	@staticmethod
	def short_str():
		return "a2c"
	# to train RL agents  on an envrionment
	@staticmethod
	def train_agents_on_env(agents, env, max_episodes: int = 100, max_steps_per_episode: int = TOTAL_TIME_STEPS, batch_percentage: float = 0.032):
		# Run the model for E episodes
		T = int(max_steps_per_episode*batch_percentage)
		completed_steps = T
		with tqdm.trange(max_episodes) as t:
			for episode in t:
				initial_state = set_training_env(env)
				state = initial_state
				episode_reward = 0
				while completed_steps < max_steps_per_episode:
					tragectory_reward, state = train_actor_critic(state, agents, gamma=0.9, max_steps=T)
					completed_steps += T
					episode_reward += tragectory_reward

				t.set_description(f'Episode {episode}')
				print(episode_reward)
		return agents


# optimizer to apply the gradient change
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# huber loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# combineed AC loss (with simple advantage)
def compute_combined_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
	returns = tf.expand_dims(tf.tile(returns, tf.constant([1,N_NODES], tf.int32)), 1)
	advantages = returns-values
	action_log_probs = tf.math.log(action_probs)
	print(advantages.shape, action_log_probs.shape)
	# 
	actor_loss = -tf.math.reduce_sum(action_log_probs * advantages, [0,1]) # for 5 actors
	critic_loss = huber_loss(values, returns)

	return actor_loss + critic_loss

# to train the advantage actor critic algorithm
#@tf.function
def train_actor_critic(initial_state: tf.Tensor, agents: List[tf.keras.Model],
	optimizer: tf.keras.optimizers.Optimizer = optimizer, gamma: float = 0.99,
	max_steps: int = 10):

	with tf.GradientTape(persistent=True) as tape:
		
		action_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		action_probs, values, rw, state = run_tragectory(initial_state, agents, max_steps)

		# Calculate simple advantage and returns
		returns = get_expected_returns(rw, gamma)

		# Convert training data to appropriate TF tensor shapes
		action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
		print(action_probs.shape, values.shape, returns.shape)
		# transpose -> expand_dims -> repeat
		
		# Calculating loss values to update our network
		loss = compute_combined_loss(action_probs, values, returns)

	print(action_loss.shape)
	# Compute the gradients from the loss for the model
	grads = tape.gradient(action_loss, [agent.model.trainable_variables for agent in agents])
	print(grads.shape)
	optimizer.apply_gradients(zip(grads,[agent.model.trainable_variables for agent in agents]))
	
	tragectory_reward = tf.math.reduce_sum(rw)
	return tragectory_reward, state


"""		for i, agent in enumerate(agents):
			# compute stuff for each agent
			aux_val = tf.identity(values[i])
			aux_action_probs = tf.identity(action_probs[i])
			
			# Calculate simple advantage and returns
			returns = get_expected_returns(rw, aux_val, gamma)

			# Convert training data to appropriate TF tensor shapes
			aux_action_probs, aux_val, returns = [tf.expand_dims(x, 1) for x in [aux_action_probs, aux_val, returns]] 

			# Calculating loss values to update our network
			loss = compute_combined_loss(aux_action_probs, aux_val, returns)
			action_losses = action_losses.write(i, loss)
		action_losses = action_losses.stack()
		action_losses = tf.expand_dims(action_losses,1)


	for i, agent in enumerate(agents):
		action_loss = tf.gather(action_losses,1) # FIX THIS??
		print("~~ ~~~~~~~~~~",action_loss)
		# Compute the gradients from the loss for the model
		grads = tape.gradient(action_loss, agent.model.trainable_variables)
		optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))
	# drop reference to the tape
	del tape
	episode_reward = tf.math.reduce_sum(rw)
	return episode_reward, state
"""	

