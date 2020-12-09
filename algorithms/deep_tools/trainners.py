#!/usr/bin/env python

# mathematic and tensor related
import numpy as np
import tensorflow as tf
from typing import Any, List, Sequence, Tuple

from .savers import save_orchestrator_models
from .common import combined_loss, general_advantage_estimator, critic_loss, actor_loss, ppo_actor_loss
from .common import map_int_vect_to_int, map_int_to_int_vect

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
		running_action = tf.TensorArray(dtype=tf.int32, size=orchestrator.num_actors)
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

# --- the generic training function for an A2C architecture ---

optimizer = tf.keras.optimizers.SGD(learning_rate=DEFAULT_LEARNING_RATE)
optimizer_1 = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE)

def train_orchestrator_on_env(orchestrator, env, total_iterations: int = DEFAULT_ITERATIONS, trajectory_lenght: int = DEFAULT_TRAJECTORY,
	batch_size: int = DEFAULT_BATCH_SIZE, epochs: int = DEFAULT_EPOCHS, saving: bool = False):

	# return values
	iteration_rewards = []
	# set the training env
	initial_state = set_training_env(env)
	current_state = initial_state
	# Run the model for total_iterations
	for iteration in range(total_iterations):
		states, actions, rewards, dones, values, run_action_probs = run_tragectory(current_state, orchestrator, trajectory_lenght)
		print("Iterations",iteration," [iteration total reward:", tf.reduce_sum(rewards).numpy(), "]") # iteration print

		train_dataset = tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, values, run_action_probs))
		train_dataset = train_dataset.shuffle(buffer_size=trajectory_lenght+batch_size).batch(batch_size)

		for e in tf.range(epochs):
			losses = {"critic": 0, 0:0, 1:0, 2:0, 3:0, 4:0}
			for state, action, rw, done, v, old_action_probs in train_dataset:
				advantages, target_values = general_advantage_estimator(rw[:-1], v[:-1], v[1:], done[1:], DEFAULT_GAMMA)

				# train the critic
				joint_state = tf.reshape(state,  [tf.shape(state)[0], tf.shape(state)[1]*tf.shape(state)[2]]) # keep batch, merge nodes
				with tf.GradientTape() as tape:
					values = orchestrator.critic(joint_state, training=True)
					values = tf.squeeze(values)
					loss = critic_loss(values[:-1], target_values)
				grads = tape.gradient(loss, orchestrator.critic.trainable_weights)
				optimizer.apply_gradients(zip(grads, orchestrator.critic.trainable_weights))
				del tape

				losses["critic"] += loss.numpy()

				#and each actor
				for i in tf.range(orchestrator.num_actors):
					with tf.GradientTape() as tape:
						# 
						action_logits = orchestrator.actors[i](state[:,i], training=True)

						# for every discrete action ~ change to probs and organize it by batches
						action_probs = tf.TensorArray(dtype=tf.float32, size=DEFAULT_BATCH_SIZE)
						for t in tf.range(DEFAULT_BATCH_SIZE):
							action_probs = action_probs.write(t, tf.nn.softmax(action_logits[t])[action[t,i]])
						action_probs = action_probs.stack()

						loss = ppo_actor_loss(old_action_probs[:-1,i], action_probs[:-1], advantages)
					grads = tape.gradient(loss, orchestrator.actors[i].trainable_weights)
					optimizer.apply_gradients(zip(grads, orchestrator.actors[i].trainable_weights))
					del tape

					losses[i.numpy()] += loss.numpy()

			print("[EPOCH",e.numpy()+1,"/",epochs,"] cumulative losses:", losses) # epoch print

		# reset the env if needed
		if training_env.clock + trajectory_lenght*TIME_STEP >= SIM_TIME:
			training_env.reset()
		current_state = training_env._get_state_obs()
		# saving values
		iteration_rewards.append(tf.reduce_sum(rewards).numpy())

	# save trained orchestrator, then return it
	if saving: save_orchestrator_models(orchestrator)
	return orchestrator, iteration_rewards
