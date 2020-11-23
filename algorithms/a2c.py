#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame, Actor_Critic_Output_Frame
from algorithms.deep_tools.common import get_expected_returns
from algorithms.runners import run_episode, set_training_env

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE
from sim_env.configs import N_NODES

# and external imports
import numpy as np
import tensorflow as tf
import tqdm
from typing import Tuple, List


# the class itself
class A2C_Agent(object):
	"""A2C_Agent
	"""
	def __init__(self, n):
		super(A2C_Agent, self).__init__()
		action_possibilities = np.append([N_NODES+1 for _ in range(n.max_k)],
			[min(n._avail_cpu_units, n._avail_ram_units)+1 for _ in range(n.max_k)])
		action_possibilities = np.array(action_possibilities, dtype=np.uint8)
		# actual agent - the NN
		self.input_model = Simple_Frame()
		self.output_model = Actor_Critic_Output_Frame(action_possibilities)
		
		# meta-data
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
		return action_i

	def model(self, obs):
		return self.output_model(self.input_model(obs))


# optimizer to apply the gradient change
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# huber loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# combineed AC loss (with simple advantage)
def compute_combined_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
	advantages = returns-values
	action_log_probs = tf.math.log(action_probs)
	actor_loss = -tf.math.reduce_sum(action_log_probs * advantages, 1)
	critic_loss = huber_loss(values, returns)

	return actor_loss + critic_loss

# to train the advantage actor critic algorithm
@tf.function
def train_actor_critic(initial_state: tf.Tensor, agents: List[tf.keras.Model],
	optimizer: tf.keras.optimizers.Optimizer = optimizer, gamma: float = 0.99,
	max_steps: int = 10):

	with tf.GradientTape(persistent=True) as tape:
		
		losses = []
		action_probs, values, rewards = run_episode(initial_state, agents, max_steps)

		for i, agent in enumerate(agents):
			# compute stuff for each agent
			aux_val = tf.identity(values[i])
			aux_action_probs = tf.identity(action_probs[i])
			
			# Calculate simple advantage and returns
			returns = get_expected_returns(rewards[i],aux_val, gamma)

			# Convert training data to appropriate TF tensor shapes
			aux_action_probs, aux_val, returns = [tf.expand_dims(x, 1) for x in [aux_action_probs, aux_val, returns]] 

			# Calculating loss values to update our network
			loss = compute_combined_loss(aux_action_probs, aux_val, returns)
			losses.append(loss)

	# TODO@luis: split losses in an approriate way for the gradient L = (100, 6)
	for agent, loss in zip(agents, losses):
		# Compute the gradients from the loss
		grads1 = tape.gradient(loss, agent.input_model.trainable_variables)
		grads2 = tape.gradient(loss, agent.output_model.trainable_variables)
		# Apply the gradients to the model's parameters
		optimizer.apply_gradients(zip(grads1, agent.input_model.trainable_variables))
		optimizer.apply_gradients(zip(grads2, agent.output_model.trainable_variables))
	# drop reference to the tape
	del tape

	episode_reward = tf.math.reduce_sum(tf.transpose(rewards),0)
	return episode_reward

# to train RL agents  on an envrionment
def run_a2c_agents_on_env(agents, env, case, max_episodes: int = 100):
	running_reward = 0
	reward_threshold = 10000
	max_steps_per_episode = 1000
	# Run the model for one episode to collect training data
	with tqdm.trange(max_episodes) as t:
		for i in t:
			initial_state = set_training_env(env)
			episode_reward = train_actor_critic(initial_state, agents, gamma=0.9, max_steps=max_steps_per_episode)

			t.set_description(f'Episode {i}')
			#t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
			print(sum(episode_reward))

			if sum(episode_reward) > reward_threshold:  
				break
	return agents