#!/usr/bin/env python

# mathematic and tensor related
import numpy as np
import tensorflow as tf
import tqdm
from typing import Any, List, Sequence, Tuple

from .savers import save_agent_models
from .common import normalize_state

# constants
from sim_env.configs import TIME_STEP, SIM_TIME
from algorithms.configs import ALGORITHM_SEED, DEFAULT_ITERATIONS, DEFAULT_TRAJECTORY, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE
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
	"""Runs a single tragectory to collect training data for each agent."""

	action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
	values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	dones = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

	initial_state_shape = initial_state.shape
	state = initial_state

	# then collect the data
	for t in tf.range(max_steps):
		# needed vars to stack agents on each timestep
		state_t = tf.TensorArray(dtype=tf.float32, size=len(agents))
		action_t = tf.TensorArray(dtype=tf.int32, size=len(agents))
		action_probs_t =  tf.TensorArray(dtype=tf.float32, size=len(agents))
		values_t = tf.TensorArray(dtype=tf.float32, size=len(agents))

		# Run every agent
		for i, agent in enumerate(agents):
			# Convert state into a batched tensor (batch size = 1)
			state_t_i = normalize_state(state[i], agent.observation_space_max)
			state_t_i = tf.expand_dims(state_t_i, 0)
			state_t = state_t.write(i, state_t_i)

			# Run the model and to get action probabilities and critic value
			model_output = agent.model(state_t_i)
			value = model_output[-1]
			action_logits_t_i = model_output[:-1]

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

	# Stack them for every time step
	action_probs = action_probs.stack()
	actions =actions.stack()
	states = states.stack()
	values = values.stack()
	# and re make them for struct [agent, time_steps, [default_size]]
	action_probs_ret_val =  tf.TensorArray(dtype=tf.float32, size=len(agents))
	actions_ret_val = tf.TensorArray(dtype=tf.int32, size=len(agents))
	states_ret_val = tf.TensorArray(dtype=tf.float32, size=len(agents))
	values_ret_val = tf.TensorArray(dtype=tf.float32, size=len(agents))
	for i in tf.range(len(agents)):
		action_probs_ret_val = action_probs_ret_val.write(i, action_probs[:,i])
		actions_ret_val = actions_ret_val.write(i, actions[:,i])
		states_ret_val = states_ret_val.write(i, states[:,i])
		values_ret_val = values_ret_val.write(i, values[:,i])
	action_probs_ret_val = action_probs_ret_val.stack()
	actions_ret_val = actions_ret_val.stack()
	states_ret_val = states_ret_val.stack()
	values_ret_val = values_ret_val.stack()
	# here both are common for every agent
	rewards = rewards.stack()
	dones = dones.stack()

	return states_ret_val, action_probs_ret_val, actions_ret_val, values_ret_val, rewards, dones

# --- the generic training function for an A2C architecture ---

def train_agents_on_env(agents, env, total_iterations: int = DEFAULT_ITERATIONS, trajectory_lenght: int = DEFAULT_TRAJECTORY,
	batch_size: int = DEFAULT_BATCH_SIZE, epochs: int = DEFAULT_EPOCHS, saving: bool = True):
	try:
		assert trajectory_lenght > batch_size # B <= N*T (N=1, parallel agents on the same node)
	except Exception as e:
		raise InvalidValueError("Batch size has to be smaller and able to divide an trajectory length")

	# set the training env
	initial_state = set_training_env(env)
	current_state = initial_state
	# Run the model for total_iterations
	with tqdm.trange(total_iterations) as t:
		for iteration in t:
			# run the trajectory
			states, action_probs, actions, values, rw, dones = run_tragectory(current_state, agents, trajectory_lenght)
			# reset the env if needed
			if training_env.clock + trajectory_lenght*TIME_STEP >= SIM_TIME:
				training_env.reset()
			current_state = training_env._get_state_obs()

			# and apply training steps for each agent
			for i, agent in enumerate(agents):
				# shared reward!
				agent.train(states[i], action_probs[i], actions[i], values[i], rw, dones, batch_size, epochs)
			print(current_state)
			t.set_description(f'Iteration {iteration}')
			print(tf.reduce_sum(rw)) # episode reward

	# save trained agents, then return them
	if saving:
		for agent in agents:
			save_agent_models(agent)
	return agents
