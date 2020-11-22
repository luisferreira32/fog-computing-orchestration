#!/usr/bin/env python

# external imports
import numpy as np
import copy
import tensorflow as tf
from typing import Tuple, List
# env imports
from sim_env.envrionment import Fog_env
# and runners?

# near 0 number
eps = np.finfo(np.float32).eps.item()

# General Advantage Estimator
def get_gaes(rewards: tf.Tensor, values: tf.Tensor, next_values: tf.Tensor, lbd: float,
	gamma: float, standardize: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
	"""Compute the general advantage estimation per timestep
	"""
	deltas = [r + gamma * nv - v for r, nv, v in zip(rewards, next_values, values)]
	deltas = np.stack(deltas)
	gaes = copy.deepcopy(deltas)
	for t in reversed(range(len(deltas) - 1)):
		gaes[t] = gaes[t] + gamma * lbd * gaes[t + 1]

	returns = gaes + values
	if standardize:
		gaes = (gaes - gaes.mean()) / (gaes.std() + eps)

	# advantage and expected returns Q(s,a)
	return tf.convert_to_tensor(gaes), tf.convert_to_tensor(returns)

# Or just the basic
def get_simple_advantage(rewards: tf.Tensor, values: tf.Tensor,
	gamma: float, standardize: bool = True) -> tf.Tensor:
	"""Compute expected returns per timestep with just discounted sum
	G_t = sum(from t'=t to T){ gamma^(t'-t) * r_t'}
	"""

	n = tf.shape(rewards)[0]
	returns = tf.TensorArray(dtype=tf.float32, size=n)

	# Start from the end of `rewards` and accumulate reward sums
	# into the `returns` array
	rewards = tf.cast(rewards[::-1], dtype=tf.float32)
	discounted_sum = tf.constant(0.0)
	discounted_sum_shape = discounted_sum.shape
	for i in tf.range(n):
		reward = rewards[i]
		discounted_sum = reward + gamma * discounted_sum
		discounted_sum.set_shape(discounted_sum_shape)
		returns = returns.write(i, discounted_sum)
	returns = returns.stack()[::-1]

	if standardize:
		returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

	# advantage and expected returns Q(s,a)
	return returns-values, returns


def tensor_list(l: List) -> List[tf.Tensor]:
	return [tf.convert_to_tensor(item) for item in l]


def run_episode(env: Fog_env, agents: List[tf.keras.Model], max_steps: int) -> List[List[tf.Tensor]]:
	"""Runs a single episode to collect training data."""

	action_probs = [tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True) for _ in agents]
	values = [tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True) for _ in agents]
	rewards = [tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True) for _ in agents]

	initial_state = tf.constant(env.reset(), dtype=tf.uint8)
	initial_state_shape = initial_state.shape
	state = initial_state

	for t in tf.range(max_steps):
		# Create the action
		action = []

		# Run every agent
		for i, agent in enumerate(agents):
			# Convert state into a batched tensor (batch size = 1)
			state_i = tf.expand_dims(state[i], 0)

			# Run the model and to get action probabilities and critic value
			retv = agent.model(state_i)
			if len(retv) == 2:
				action_logits_t, value = retv
			else:
				action_logits_t = retv; value = 0.0

			# Get the action and probability distributions for data
			action_i = []; action_probs_t = []
			# Since it's multi-discrete, for every discrete set of actions:
			for action_logits_t_k in action_logits_t:
				# Sample next action from the action probability distribution
				action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
				action_i.append(action_i_k)
				action_probs_t_k = tf.nn.softmax(action_logits_t_k)
				action_probs_t.append(action_probs_t_k[0, action_i_k])

			# Store critic values
			values[i] = values[i].write(t, tf.squeeze(value))
			# Store log probability of the action chosen
			action_probs[i] = action_probs[i].write(t, action_probs_t)

			# And append to the actual action that is gonna run
			action.append(action_i)

		# Apply action to the environment to get next state and reward
		state, reward, done, _ = env.step(np.array(action))
		state, reward, done = tensor_list([state, reward, done])
		state.set_shape(initial_state_shape)
		#print(state, reward, done, action)
		# Store reward
		for i in range(len(agents)):
			rewards[i] = rewards[i].write(t, reward[i])

		if tf.cast(done, tf.bool):
			break

	# Stack them for every agent
	for i in range(len(agents)):
		action_probs[i] = action_probs[i].stack()
		values[i] = values[i].stack()
		rewards[i] = rewards[i].stack()

	return action_probs, values, rewards


# and adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# to train an actor critic algorithm
@tf.function
def train_actor_critic(action_probs, values, rewards, agents: List[tf.keras.Model],
	optimizer: tf.keras.optimizers.Optimizer = optimizer,
	advantage_calculator = get_simple_advantage, gamma: float = 0.99):

	for agent, a_probs, v, rw in zip(agents,action_probs,values,rewards):
		with tf.GradientTape() as tape:
			tape.watch(a_probs)
			tape.watch(v)
			tape.watch(rewards)

			# Calculate simple advantage and returns
			adv, returns = advantage_calculator(rw,v, gamma)

			# Convert training data to appropriate TF tensor shapes
			a_probs, v, adv, returns = [tf.expand_dims(x, 1) for x in [a_probs, v, adv, returns]] 

			# Calculating loss values to update our network
			loss = agent.compute_combined_loss(a_probs, v, adv, returns)

		# Compute the gradients from the loss
		grads = tape.gradient(loss, agent.model.trainable_variables)
		print(grads.shape)

		# Apply the gradients to the model's parameters
		optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))

	episode_reward = tf.math.reduce_sum(rewards)

	return episode_reward
