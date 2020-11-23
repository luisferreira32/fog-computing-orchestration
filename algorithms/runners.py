#!/usr/bin/env python

# for information gathering
from utils.tools import dictionary_append, append_to_file
from utils.display import info_gather, info_logs

# the envrionment
from sim_env.envrionment import Fog_env
from algorithms.deep_tools.common import  set_tf_seed

# constants
from algorithms.configs import ALGORITHM_SEED

# mathematic and tensor related
import numpy as np
import time
import tensorflow as tf
import tqdm
from typing import Any, List, Sequence, Tuple
import os


# -- baseline related runners --

def run_algorithm_on_envrionment(agents, env, case, compiled_info=None, debug=False):
	# runner for simple baseline algorithms
	start_time = time.time()
	obs_n = env.reset()
	done = False;
	while not done:
		action_n = np.array([agent(obs) for agent,obs in zip(agents, obs_n)], dtype=np.uint8)
		obs_n, rw_n, done, info_n = env.step(action_n)
		if debug: env.render()
		# -- info gathering
		if compiled_info is not None: compiled_info = info_gather(compiled_info, info_n)
		# --

	# -- info logs
	if compiled_info is not None: info_logs(str(agents[0])+str(case), round(time.time()-start_time,2), compiled_info)
	# --
	return compiled_info


# --- training related running ---

#
def create_and_train_agents(env, alg, case):
	set_tf_seed(ALGORITHM_SEED)
	# save path for the models
	my_path = os.getcwd() + "/algorithms/saved_models/"+alg.short_str()+case["case"]+"/"
	# the agents and run them for training
	agents = [alg() for n in env.nodes]
	agents = alg.run_agents_on_env(agents, env)
	for agent in agents:
		agent.save_models(my_path)
	return agents

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

def run_episode(initial_state: tf.Tensor, agents: List[tf.keras.Model], max_steps: int) -> List[List[tf.Tensor]]:
	"""Runs a single episode to collect training data."""

	action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

	initial_state_shape = initial_state.shape
	state = initial_state

	# run a couple times the env without collecting? TODO@luis

	# then collect the data
	for t in tf.range(max_steps):
		# needed vars
		action = []; value = [];
		action_probs_v = []; v = 0

		# Run every agent
		for i, agent in enumerate(agents):
			# Convert state into a batched tensor (batch size = 1)
			state_i = tf.expand_dims(state[i], 0)

			# Run the model and to get action probabilities and critic value
			retv = agent.model(state_i)
			if len(retv) == 2:
				action_logits_t, v = retv
			else:
				action_logits_t = retv

			# Get the action and probability distributions for data
			action_i = []; action_probs_t = []
			# Since it's multi-discrete, for every discrete set of actions:
			for action_logits_t_k in action_logits_t:
				# Sample next action from the action probability distribution
				action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
				action_i.append(action_i_k)
				action_probs_t_k = tf.nn.softmax(action_logits_t_k)
				action_probs_t.append(action_probs_t_k[0, action_i_k])

			# And append to the actual action that is gonna run
			action.append(action_i)
			value.append(v)
			action_probs_v.append(action_probs_t)
		# Store critic values
		values = values.write(t, tf.squeeze(value))
		# Store log probability of the action chosen
		action_probs = action_probs.write(t, action_probs_v)


		# Apply action to the environment to get next state and reward
		state, reward, done = tf_env_step(action)
		state.set_shape(initial_state_shape)

		# Store reward
		rewards = rewards.write(t, reward)

		if tf.cast(done, tf.bool):
			break

	# Stack them for every agent and set struct: shape=[agents, time_steps,[vars]]
	action_probs = tf.transpose(action_probs.stack(), perm=[1,0,2])
	values = tf.transpose(values.stack())
	rewards = tf.transpose(rewards.stack())

	return action_probs, values, rewards

