#!/usr/bin/env python

## IMPORTANT NOTE: this algorithm was not actually extensively tested, might be bugged...

import tensorflow as tf
import numpy as np
import time

from algorithms.deep_tools.frames import Simple_Frame, Frame_1
from algorithms.deep_tools.common import map_int_vect_to_int, map_int_to_int_vect, roll_one_step, critic_loss

from algorithms.configs import INITIAL_EPSILON, MIN_EPSILON, EPSILON_RENEWAL_RATE, EPSILON_RENEWAL_FACTOR, ALGORITHM_SEED
from algorithms.configs import DEFAULT_BATCH_SIZE, TARGET_NETWORK_UPDATE_RATE, MAX_DQN_TRAIN_ITERATIONS, DEFAULT_GAMMA
from algorithms.configs import REPLAY_BUFFER_SIZE, TIME_SEQUENCE_SIZE, DEFAULT_DQN_LEARNING_RATE
from algorithms.configs import DEFAULT_SAVE_MODELS_PATH

class Dqn_Orchestrator(object):
	"""docstring for Dqn_Orchestrator"""
	basic = False
	def __init__(self, env, dqn_frame = Frame_1):
		super(Dqn_Orchestrator, self).__init__()

		# N DQN for each node; and N DQN target for each node too
		self.actors_dqn =  [dqn_frame(map_int_vect_to_int(action_space_n)+1, 11) for action_space_n in env.action_space.nvec]
		self.num_actors = len(env.nodes)

		# meta-data
		self.name = env.case["case"]+"_rd"+str(env.rd_seed)+"_dqn_orchestrator_"+Frame_1.short_str()
		self.env = env
		self.action_spaces = env.action_space.nvec
		self.observation_spaces = env.observation_space.nvec

		self.epsilon = INITIAL_EPSILON
		self.policy_random = np.random.RandomState(seed=ALGORITHM_SEED)

		# and aux variables
		self.act_state = None

	def __str__(self):
		return self.name

	@staticmethod
	def short_str():
		return "dqn"

	def policy(self, q_values):
		# returns an action depending on the current policy and given Q-values
		r = self.policy_random.rand()
		if r < self.epsilon: # random action
			action_int = self.policy_random.randint(len(q_values))
		else: # maximizing action
			action_int = tf.math.argmax(q_values)
		return action_int

	def act(self, obs_n):
		""" takes the whole system observation and returns the action step of all the nodes """
		if self.act_state is None:
			x = tf.expand_dims(obs_n, 0)
			self.act_state = tf.repeat(x, repeats=TIME_SEQUENCE_SIZE, axis=0)
		# roll one timestep and stack the new obs
		self.act_state = roll_one_step(self.act_state, obs_n)

		# for each agent decide an action
		action = []
		for i in range(self.num_actors):
			obs = self.act_state[:,i]
			actor_dqn = self.actors_dqn[i]
			action_space = self.action_spaces[i]
			# just one batch
			obs = tf.expand_dims(obs, 0)
			# call its model
			q_values = actor_dqn(obs)
			# remove the batch and calculate the action
			action_i = map_int_to_int_vect(action_space, self.policy(tf.squeeze(q_values)))
			action.append(action_i)
		return np.array(action)

	def save_models(self, saved_models_path=DEFAULT_SAVE_MODELS_PATH):
		# function to save the models of this algorithm
		complete_path = saved_models_path + self.name
		for i, model in enumerate(self.actors_dqn):
			model.save(complete_path+"_node"+str(i))

	def load_models(self, saved_models_path=DEFAULT_SAVE_MODELS_PATH):
		# try to load the saved models
		complete_path = saved_models_path + self.name
		load_fun = tf.keras.models.load_model
		try:
			for i in range(self.num_actors):
				self.actors_dqn[i] = load_fun(complete_path+"_node"+str(i), compile=False)
			self.update_target_network()
		except Exception as e:
			print("[ERROR LOG] It was not able to load the specified orchestrator model from", saved_models_path)

	def update_target_network(self, actors_dqn_target):
		for dqn_target, dqn in zip(actors_dqn_target, self.actors_dqn):
			dqn_target.set_weights(dqn.get_weights())

	def train(self, batch_size: int = DEFAULT_BATCH_SIZE):
		# init a replay buffer with fixed size
		replay_buffer_states = tf.TensorArray(dtype=tf.float32, size=REPLAY_BUFFER_SIZE)
		replay_buffer_actions = tf.TensorArray(dtype=tf.float32, size=REPLAY_BUFFER_SIZE)
		replay_buffer_rewards = tf.TensorArray(dtype=tf.float32, size=REPLAY_BUFFER_SIZE)
		replay_buffer_next_states = tf.TensorArray(dtype=tf.float32, size=REPLAY_BUFFER_SIZE)

		# run for 10^4 iterations from random sampling to the replay buffer
		obs_n = self.env.reset()
		x = tf.expand_dims(obs_n, 0)
		state = tf.repeat(x, repeats=TIME_SEQUENCE_SIZE, axis=0)

		print("[LOG] Starting replay buffer", flush=True)
		for t in tf.range(REPLAY_BUFFER_SIZE):

			# pick up the action
			action = []; action_integers = [];
			for i in range(self.num_actors):
				obs = state[:,i]
				actor_dqn = self.actors_dqn[i]
				action_space = self.action_spaces[i]
				# just one batch
				obs = tf.expand_dims(obs, 0)
				# call its model
				q_values = actor_dqn(obs)
				# remove the batch and calculate the action
				action_i = self.policy(tf.squeeze(q_values))
				action.append(map_int_to_int_vect(action_space, action_i))
				action_integers.append(action_i)
			obs_n, reward, done, _ = self.env.step(np.array(action))

			# roll a time_step and concat observation
			next_state = roll_one_step(state, obs_n)

			replay_buffer_states = replay_buffer_states.write(t, state)
			replay_buffer_actions = replay_buffer_actions.write(t, action_integers)
			replay_buffer_rewards = replay_buffer_rewards.write(t, reward)
			replay_buffer_next_states = replay_buffer_next_states.write(t, next_state)

			state = tf.identity(next_state)
			if t%1000 == 0:
				print(".", end='', flush=True)
		print(" ")
		print("[LOG] Done starting replay buffer!", flush=True)

		replay_buffer_states = replay_buffer_states.stack()
		replay_buffer_actions = replay_buffer_actions.stack()
		replay_buffer_rewards = replay_buffer_rewards.stack()
		replay_buffer_next_states = replay_buffer_next_states.stack()
		
		# set up a target network
		actors_dqn_target =  [Frame_1(map_int_vect_to_int(action_space_n)+1, 11) for action_space_n in self.action_spaces]
		for dqn_target in actors_dqn_target:
			obs = state[:,0]
			obs = tf.expand_dims(obs, 0)
			dqn_target(obs)
		self.update_target_network(actors_dqn_target)

		# train the DQN agents
		# set up variables
		iter_rewards = []
		optimizer = tf.keras.optimizers.Adam(learning_rate=DEFAULT_DQN_LEARNING_RATE)
		huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

		# for epsilon calc too
		e_start = INITIAL_EPSILON
		R_e = EPSILON_RENEWAL_RATE
		e_decay = (e_start - MIN_EPSILON)/R_e

		for t in tf.range(1, MAX_DQN_TRAIN_ITERATIONS):
			start_time = time.time()
			# update epsilon
			self.epsilon = max(e_start-e_decay*(tf.cast(t%R_e, tf.float32)), MIN_EPSILON)

			# run with last obtained state:
			# 1. pick up the action
			action = []; action_integers = [];
			for i in range(self.num_actors):
				obs = state[:,i]
				actor_dqn = self.actors_dqn[i]
				action_space = self.action_spaces[i]
				# just one batch
				obs = tf.expand_dims(obs, 0)
				# call its model
				q_values = actor_dqn(obs)
				# remove the batch and calculate the action
				action_i = self.policy(tf.squeeze(q_values))
				action.append(map_int_to_int_vect(action_space, action_i))
				action_integers.append(action_i)
			obs_n, reward, done, _ = self.env.step(np.array(action))
			iter_rewards.append(reward)

			# 2. roll a time_step and concat observation
			next_state = roll_one_step(state, obs_n)

			# 3. save new iteration
			replay_buffer_states = roll_one_step(replay_buffer_states, state)
			replay_buffer_actions = roll_one_step(replay_buffer_actions, action_integers)
			replay_buffer_rewards = roll_one_step(replay_buffer_rewards, reward)
			replay_buffer_next_states = roll_one_step(replay_buffer_next_states, next_state)

			# 4. and
			state = tf.identity(next_state)


			# sample random mini-batch and perform gradients to update network
			# 1. create a dataframe and shuffle it in batches
			train_dataset = tf.data.Dataset.from_tensor_slices((replay_buffer_states, replay_buffer_actions, replay_buffer_rewards, replay_buffer_next_states))
			train_dataset = train_dataset.shuffle(buffer_size=REPLAY_BUFFER_SIZE+REPLAY_BUFFER_SIZE).batch(batch_size)

			# 2. sample just one mini batch
			for data in train_dataset:
				(train_state, train_action, train_reward, train_next_state) = data
				break
			#print(train_state.shape, train_next_state.shape, train_reward.shape, train_action.shape)

			# 3. compute targets and gradeient for every actor, then apply it with optimizer 
			for i in tf.range(self.num_actors):
				actor_train_state = train_state[:,:,i]
				actor_train_next_state = train_next_state[:,:,i]

				y_target = tf.TensorArray(dtype=tf.float32, size=batch_size)
				for b_t in tf.range(batch_size):
					x = tf.expand_dims(actor_train_next_state[b_t], 0)
					q_next_target_values = actors_dqn_target[i](x)
					q_next_target_values = tf.squeeze(q_next_target_values)
					y_target = y_target.write(b_t, train_reward[b_t] + DEFAULT_GAMMA*tf.reduce_max(q_next_target_values))
				y_target = y_target.stack()
				#print(y_target.shape)
			
				with tf.GradientTape() as tape:
					q_values = self.actors_dqn[i](actor_train_state)

					# pick up actual chosen Q values
					actual_q_values =  tf.TensorArray(dtype=tf.float32, size=batch_size)
					for b_t in tf.range(batch_size):
						actual_q_values = actual_q_values.write(b_t, q_values[b_t][int(train_action[b_t, i])])
					actual_q_values = actual_q_values.stack()

					loss = huber_loss(actual_q_values, y_target)
				grads = tape.gradient(loss, self.actors_dqn[i].trainable_weights)
				optimizer.apply_gradients(zip(grads, self.actors_dqn[i].trainable_weights))

			# every C steps replace target network with trained network
			if t%TARGET_NETWORK_UPDATE_RATE == 0:
				self.update_target_network(actors_dqn_target)

			# update epsilon updating parameters every R_e
			if t%R_e == 0:
				e_start = e_start*EPSILON_RENEWAL_FACTOR
				e_decay = (e_start - MIN_EPSILON)/R_e

			if t%10 == 0:
				print("Iteration",t.numpy()," [iteration reward:", reward, "] [time", round(time.time()-start_time,2),"]", flush=True) # iteration print

		return iter_rewards
		