#!/usr/bin/env python

# mathematic and tensor related
import numpy as np
import tensorflow as tf
from typing import Any, List, Sequence, Tuple

from .savers import save_orchestrator_models
from .common import combined_loss, general_advantage_estimator, critic_loss, actor_loss

# constants
from sim_env.configs import TIME_STEP, SIM_TIME
from algorithms.configs import ALGORITHM_SEED, DEFAULT_ITERATIONS, DEFAULT_TRAJECTORY, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE
from algorithms.configs import DEFAULT_GAMMA, DEFAULT_LEARNING_RATE
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

def run_tragectory(initial_state: tf.Tensor, orchestrator, max_steps: int) -> List[tf.Tensor]:
	"""Runs a single tragectory to collect training data for each agent."""

	action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
	actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
	values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	dones = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

	initial_state_shape = initial_state.shape
	state = initial_state

	# then collect the data
	for t in tf.range(max_steps):
		# needed vars to stack agents on each timestep
		state_t = tf.TensorArray(dtype=tf.uint8, size=orchestrator.num_actors)
		action_t = tf.TensorArray(dtype=tf.int32, size=orchestrator.num_actors)
		action_probs_t =  tf.TensorArray(dtype=tf.float32, size=orchestrator.num_actors)

		# obtain common critic value
		reshaped_state = tf.reshape(state, [tf.shape(state)[0]*tf.shape(state)[1]])
		reshaped_state = tf.expand_dims(reshaped_state, 0) # batch size = 1
		value = orchestrator.critic(reshaped_state)
		values = values.write(t, tf.squeeze(value)) # squeeze it out of batches

		# Run every agent
		for i in tf.range(orchestrator.num_actors):
			state_t_i = tf.expand_dims(state[i], 0) # batch size = 1
			state_t = state_t.write(i, state_t_i)

			# Run the model and to get action probabilities and critic value
			action_logits_t_i = orchestrator.actors[i](state_t_i)

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

		# stack agents state, action pair this timestep
		action_t = action_t.stack()
		# Stack and store timestep values
		action_probs = action_probs.write(t, action_probs_t.stack())
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
	# and re make them for struct [agent, time_steps, [default_size]]
	actions_ret_val = tf.TensorArray(dtype=tf.int32, size=orchestrator.num_actors)
	states_ret_val = tf.TensorArray(dtype=tf.uint8, size=orchestrator.num_actors)
	for i in tf.range(orchestrator.num_actors):
		actions_ret_val = actions_ret_val.write(i, actions[:,i])
		states_ret_val = states_ret_val.write(i, states[:,i])
	actions_ret_val = actions_ret_val.stack()
	states_ret_val = states_ret_val.stack()
	# here all are common for every agent
	rewards = rewards.stack()
	values = values.stack()
	dones = dones.stack()

	return states_ret_val, states, actions_ret_val, values, rewards, dones

# --- the generic training function for an A2C architecture ---

optimizer = tf.keras.optimizers.SGD(learning_rate=DEFAULT_LEARNING_RATE)
optimizer_1 = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE)

def train_orchestrator_on_env(orchestrator, env, total_iterations: int = DEFAULT_ITERATIONS, trajectory_lenght: int = DEFAULT_TRAJECTORY,
	batch_size: int = DEFAULT_BATCH_SIZE, epochs: int = DEFAULT_EPOCHS, saving: bool = True):

	# set the training env
	initial_state = set_training_env(env)
	current_state = initial_state
	# Run the model for total_iterations
	for iteration in range(total_iterations):
		states_n, states, actions, values, rw, dones = run_tragectory(initial_state, orchestrator, trajectory_lenght)
		advantages, target_values = general_advantage_estimator(rw[:-1], values[:-1], values[1:], dones[1:], 0.99)

		# train each actor
		losses = {}
		for i in tf.range(orchestrator.num_actors):
			with tf.GradientTape() as tape:
				# 
				action_logits = orchestrator.actors[i](states_n[i])

				# for every discrete action
				action_probs = tf.TensorArray(dtype=tf.float32, size=6) # TODO@luis: 6 it's the discrete actions ~ soft code this later
				for k in tf.range(6): # TODO@luis: 6 it's the discrete actions ~ soft code this later
					action_probs_t = tf.TensorArray(dtype=tf.float32, size=trajectory_lenght)
					for t in tf.range(trajectory_lenght):
						#print(tf.nn.softmax(action_logits[k][t])[actions[i,t,k]])
						action_probs_t = action_probs_t.write(t, tf.nn.softmax(action_logits[k][t])[actions[i,t,k]])
					action_probs = action_probs.write(k, action_probs_t.stack())
				action_probs = action_probs.stack()
				#print(action_probs)

				loss = actor_loss(action_probs[:,:-1], advantages)
			grads = tape.gradient(loss, orchestrator.actors[i].trainable_weights)
			optimizer.apply_gradients(zip(grads, orchestrator.actors[i].trainable_weights))
			losses[i.numpy()] = tf.identity(loss).numpy()
			del tape		

		# and train the critic
		with tf.GradientTape() as tape:
			batch = tf.reshape(states, [tf.shape(states)[0], tf.shape(states)[1]*tf.shape(states)[2]])
			values = orchestrator.critic(batch)[0]
			loss = critic_loss(values[:-1,0], target_values)
		grads = tape.gradient(loss, orchestrator.critic.trainable_weights)
		optimizer.apply_gradients(zip(grads, orchestrator.critic.trainable_weights))
		losses["critic"] = tf.identity(loss).numpy()
		del tape

		# reset the env if needed
		if training_env.clock + trajectory_lenght*TIME_STEP >= SIM_TIME:
			training_env.reset()
		current_state = training_env._get_state_obs()
		# iteration print
		print("Iterations",iteration," [iteration reward:", tf.reduce_sum(rw).numpy(), "] [losses",losses,"]")

	# save trained orchestrator, then return it
	if saving: save_orchestrator_models(orchestrator)
	return orchestrator
