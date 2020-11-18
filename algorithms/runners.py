#!/usr/bin/env python

# for information gathering
from utils.tools import dictionary_append, append_to_file
from utils.display import info_gather, info_logs

# the envrionment
from sim_env.envrionment import Fog_env

# mathematic and tensor related
import numpy as np
import time
import tensorflow as tf

# type setting
from typing import Any, List, Sequence, Tuple

# -- baseline related runners --

def run_algorithm_on_envrionment(alg, case, seed, compiled_info, debug=False):
	# runner for simple baseline algorithms
	start_time = time.time()
	env = Fog_env(case, seed)
	agents = [alg(n) for n in env.nodes]
	obs_n = env.reset()
	done = False;
	while not done:
		action_n = np.array([agent.decide(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
		obs_n, rw_n, done, info_n = env.step(action_n)
		if debug: env.render()
		# -- info gathering
		compiled_info = info_gather(compiled_info, info_n)
		# --

	# -- info logs
	info_logs(str(agents[0])+str(case), round(time.time()-start_time,2), compiled_info)
	# --
	return compiled_info

# -- RL related runners --

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
		for i in tf.range(len(agents)):
			# Pick up the model
			agent = agents[i]

			# Convert state into a batched tensor (batch size = 1)
			state_i = tf.expand_dims(state[i], 0)

			# Run the model and to get action probabilities and critic value
			action_logits_t, value = agent(state_i)

			# Sample next action from the action probability distribution
			action_i = tf.convert_to_tensor([tf.random.categorical(al, 1)[0, 0] for al in action_logits_t])
			action_probs_t = tf.nn.softmax(action_logits_t)
			action.append(action_i)

			# Store critic values
			values[i] = values[i].write(t, tf.squeeze(value))

			# Store log probability of the action chosen
			action_probs[i] = action_probs[i].write(t, action_probs_t[0, action_i])

		# Apply action to the environment to get next state and reward
		state, reward, done, _ = env.step([a.numpy() for a in action])
		state, reward, done = tensor_list([state, reward, done])
		state.set_shape(initial_state_shape)

		# Store reward
		for i in tf.range(len(agents)):
			rewards[i] = rewards[i].write(t, reward[i])

		if tf.cast(done, tf.bool):
			break

	# Stack them for every agent
	for i in tf.range(len(agents)):
		action_probs[i] = action_probs[i].stack()
		values[i] = values[i].stack()
		rewards = rewards[i].stack()

	return action_probs, values, rewards

# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
def tensor_list(l: List) -> List[tf.Tensor]:
	return [tf.convert_to_tensor(item) for item in l]
