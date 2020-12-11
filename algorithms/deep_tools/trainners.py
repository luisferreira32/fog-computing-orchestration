#!/usr/bin/env python

# mathematic and tensor related
import numpy as np
import time
import tensorflow as tf
from typing import Any, List, Sequence, Tuple

from .common import map_int_to_int_vect
from algorithms.configs import TIME_SEQUENCE_SIZE


def set_training_env(env):
	global training_env
	training_env = env
	return env.reset()

# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Returns state, reward and done flag given an action."""

	state, reward, done, _ = training_env.step(action)
	return (state.astype(np.float32), 
		np.array(reward, np.float32), 
		np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
	return tf.numpy_function(env_step, [action], 
		[tf.float32, tf.float32, tf.int32])

def run_actor_critic_tragectory(initial_state: tf.Tensor, orchestrator, max_steps: int) -> List[tf.Tensor]:
	"""Runs a single tragectory to collect training data for each agent."""

	assert np.all(initial_state == training_env._get_state_obs())
	# arrays to gather data
	action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
	values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	dones = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

	initial_state_shape = initial_state.shape
	state = initial_state

	# tensor that holds the previous state
	x = tf.expand_dims(state, 0)
	actor_state = tf.repeat(x, repeats=TIME_SEQUENCE_SIZE, axis=0)
	x = tf.expand_dims(tf.reshape(state, [tf.shape(state)[0]*tf.shape(state)[1]]), 0)
	critic_state = tf.repeat(x, repeats=TIME_SEQUENCE_SIZE, axis=0)

	# then collect the data
	for t in tf.range(max_steps):
		# needed vars to stack agents on each timestep
		state_t = tf.TensorArray(dtype=tf.float32, size=orchestrator.num_actors)
		action_t = tf.TensorArray(dtype=tf.int32, size=orchestrator.num_actors)
		running_action = tf.TensorArray(dtype=tf.int32, size=orchestrator.num_actors)
		action_probs_t =  tf.TensorArray(dtype=tf.float32, size=orchestrator.num_actors)

		# reshape the state and roll one timestep for critic
		reshaped_state = tf.reshape(state, [tf.shape(state)[0]*tf.shape(state)[1]])
		critic_state_list = tf.unstack(critic_state)
		critic_state = tf.stack((critic_state_list[1:])) # roll one timestep
		critic_state = tf.concat((critic_state, [reshaped_state]), axis=0) # and add it to the end
		critic_state = tf.expand_dims(critic_state, 0) # batch size = 1

		# obtain common critic value
		value = orchestrator.critic(critic_state)
		critic_state = tf.squeeze(critic_state) # and take out the batch for next iteration
		values = values.write(t, tf.squeeze(value)) # squeeze it out of batches

		# roll one timestep for actor
		actor_state_list = tf.unstack(actor_state)
		actor_state = tf.stack((actor_state_list[1:]))
		actor_state = tf.concat((actor_state, [state]), axis=0)

		# Run every agent
		for i in tf.range(orchestrator.num_actors):
			state_t_i = tf.expand_dims(actor_state[:,i], 0) # batch size = 1
			state_t = state_t.write(i, state_t_i)

			# Run the model and to get action probabilities for the actor
			action_logits_t_i = orchestrator.actors[i](state_t_i)
			action_t_i = tf.random.categorical(action_logits_t_i,1, dtype=tf.int32)[0,0]
			action_probs_t_i = tf.nn.softmax(action_logits_t_i)[0, action_t_i]
			running_action_i = map_int_to_int_vect(orchestrator.action_spaces[i], action_t_i.numpy())

			# And append to the actual action that is gonna run
			running_action = running_action.write(i, running_action_i)
			action_t = action_t.write(i, action_t_i)
			action_probs_t = action_probs_t.write(i, action_probs_t_i)

		# stack agents state, action pair this timestep
		running_action = running_action.stack()
		action_t = action_t.stack()
		# Stack and store timestep values
		action_probs = action_probs.write(t, action_probs_t.stack())
		states = states.write(t, tf.squeeze(state_t.stack()))
		actions = actions.write(t, action_t)


		# Apply action to the environment to get next state and reward
		state, reward, done = tf_env_step(running_action)
		state.set_shape(initial_state_shape)

		# Store reward
		rewards = rewards.write(t, reward)
		dones = dones.write(t, done)

		if tf.cast(done, tf.bool):
			break

	# Stack them for every time step
	action_probs = action_probs.stack()
	values = values.stack()
	states = states.stack()
	actions =actions.stack()
	rewards = rewards.stack()
	dones = dones.stack()

	return states, actions, rewards, dones, values, action_probs


	
