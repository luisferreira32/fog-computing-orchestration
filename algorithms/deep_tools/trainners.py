#!/usr/bin/env python

# mathematic and tensor related
import numpy as np
import tensorflow as tf
import tqdm
from typing import Any, List, Sequence, Tuple

# constants
from sim_env.configs import TOTAL_TIME_STEPS
from algorithms.configs import ALGORITHM_SEED, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_ITERATIONS
from utils.custom_exceptions import InvalidValueError


def set_training_env(env):
	global training_env
	training_env = env
	return env.reset() #env._get_state_obs()#

# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Returns state, reward and done flag given an action."""

	state, reward, done, _ = training_env.step(action)
	return (state.astype(np.uint8), 
		np.array(reward, np.float32), 
		np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
	return tf.numpy_function(env_step, [action], 
		[tf.uint8, tf.float32, tf.int32])

def run_tragectory(initial_state: tf.Tensor, agents, max_steps: int) -> List[tf.Tensor]:
	"""Runs a single tragectory to collect training data."""

	action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
	actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
	values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	dones = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

	initial_state_shape = initial_state.shape
	state = initial_state

	# run a couple times the env without collecting? TODO@luis

	# then collect the data
	for t in tf.range(max_steps):
		# needed vars to stack agents on each timestep
		state_t = tf.TensorArray(dtype=tf.uint8, size=len(agents))
		action_t = tf.TensorArray(dtype=tf.int32, size=len(agents))
		action_probs_t =  tf.TensorArray(dtype=tf.float32, size=len(agents))
		values_t = tf.TensorArray(dtype=tf.float32, size=len(agents))

		# Run every agent
		for i, agent in enumerate(agents):
			# Convert state into a batched tensor (batch size = 1)
			state_t_i = tf.expand_dims(state[i], 0)
			state_t = state_t.write(i, state_t_i)

			# Run the model and to get action probabilities and critic value
			action_logits_t_i, value = agent.model(state_t_i)

			# Get the action and probability distributions for data
			action_t_i = tf.TensorArray(dtype=tf.int32, size=len(action_logits_t_i))
			action_probs_t_i =  tf.TensorArray(dtype=tf.float32, size=len(action_logits_t_i))
			# Since it's multi-discrete, for every discrete set of actions:
			for k, action_logits_t_i_k in enumerate(action_logits_t_i):
				# Sample next action from the action probability distribution
				action_t_i_k = tf.random.categorical(action_logits_t_i_k,1, dtype=tf.int32)[0,0]
				action_t_i = action_t_i.write(k, action_t_i_k)
				action_probs_t_i_k = tf.nn.softmax(action_logits_t_i_k)
				action_probs_t_i = action_probs_t_i.write(k, action_probs_t_i_k[0, action_t_i_k])

			# And append to the actual action that is gonna run
			action_t = action_t.write(i, action_t_i.stack())
			action_probs_t = action_probs_t.write(i, action_probs_t_i.stack())
			values_t = values_t.write(i, value)

		# stack agents state, action pair this timestep
		action_t = action_t.stack()
		# Stack and store timestep values (actors log probabilities of actions chosen and critics values)
		action_probs = action_probs.write(t, action_probs_t.stack())
		values = values.write(t, tf.squeeze(values_t.stack()))
		states = states.write(t, tf.squeeze(state_t.stack()))
		actions = actions.write(t, action_t)


		# Apply action to the environment to get next state and reward
		state, reward, done = tf_env_step(action_t)
		state.set_shape(initial_state_shape)

		# Store reward
		rewards = rewards.write(t, reward)
		dones = dones.write(t, done)

		if tf.cast(done, tf.bool):
			break

	# Stack them for every agent and set struct: shape=[agents, time_steps, [particular shape]]
	action_shape = [len(agents), max_steps, len(action_logits_t_i)] # 6 distinct actions
	action_probs = tf.reshape(action_probs.stack(), action_shape)
	actions = tf.reshape(actions.stack(), action_shape)
	state_shape = [len(agents), max_steps, initial_state_shape[-1]] # 11 state values
	states = tf.reshape(states.stack(), state_shape)
	value_shape = [len(agents), max_steps] # just 1 critic value
	values = tf.reshape(values.stack(), value_shape)
	# both here are common for every agent
	rewards = rewards.stack()
	dones = dones.stack()

	return states, action_probs, actions, values, rewards, dones

# --- the generic training function for an A2C architecture ---

def train_agents_on_env(agents, env, total_iterations: int = DEFAULT_ITERATIONS, episode_max_lenght: int = TOTAL_TIME_STEPS,
	batch_size: int = DEFAULT_BATCH_SIZE, epochs: int = DEFAULT_EPOCHS):
	try:
		assert episode_max_lenght > batch_size
		assert episode_max_lenght/batch_size > 1
	except Exception as e:
		raise InvalidValueError("Batch size has to be smaller and able to divide an episode length")
		
	# Run the model for total_iterations
	with tqdm.trange(total_iterations) as t:
		for iteration in t:
			# run an episode
			initial_state = set_training_env(env)
			states, action_probs, actions, values, rw, dones = run_tragectory(initial_state, agents, episode_max_lenght)
			# and apply training steps for each agent
			for i, agent in enumerate(agents):
				# shared reward!
				agent.train(states[i], action_probs[i], actions[i], values[i], rw, dones, batch_size, epochs)

			t.set_description(f'Iteration {iteration}')
			print(tf.reduce_sum(rw)) # episode reward
	return agents
