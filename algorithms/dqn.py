#!/usr/bin/env python

## IMPORTANT NOTE: this algorithm was not actually extensively tested, might be bugged...

import tensorflow as tf
import numpy as np
import time
import math

from algorithms.deep_tools.frames import Simple_Frame, Frame_1
from algorithms.deep_tools.common import map_int_vect_to_int, map_int_to_int_vect, roll_one_step, critic_loss
from algorithms.deep_tools.dqn_tools import Replay_buffer, Temporary_experience, instant_rw_fun

from algorithms.configs import INITIAL_EPSILON, MIN_EPSILON, EPSILON_RENEWAL_RATE, EPSILON_RENEWAL_FACTOR, ALGORITHM_SEED
from algorithms.configs import DEFAULT_BATCH_SIZE, TARGET_NETWORK_UPDATE_RATE, MAX_DQN_TRAIN_ITERATIONS, DEFAULT_GAMMA
from algorithms.configs import REPLAY_BUFFER_SIZE, TIME_SEQUENCE_SIZE, DEFAULT_DQN_LEARNING_RATE
from algorithms.configs import DEFAULT_SAVE_MODELS_PATH, RW_EPS

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
		except Exception as e:
			print("[ERROR LOG] It was not able to load the specified orchestrator model from", saved_models_path)

	def update_target_network(self, actors_dqn_target):
		for dqn_target, dqn in zip(actors_dqn_target, self.actors_dqn):
			dqn_target.set_weights(dqn.get_weights())
		return actors_dqn_target


	def train(self, batch_size: int = DEFAULT_BATCH_SIZE):
		# set up training variables
		# for training steps
		instant_rw_buffer = tf.TensorArray(dtype=tf.float32, size=MAX_DQN_TRAIN_ITERATIONS)
		optimizer = tf.keras.optimizers.Adam(learning_rate=DEFAULT_DQN_LEARNING_RATE)
		huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
		# for epsilon calculations too
		e_start = INITIAL_EPSILON
		e_decay = math.log(e_start)/EPSILON_RENEWAL_RATE - math.log(MIN_EPSILON)/EPSILON_RENEWAL_RATE

		# init a replay buffer with fixed size
		experience_replay_buffer = Replay_buffer(REPLAY_BUFFER_SIZE)
		temporary_experience_buffer = []

		# set the initial state
		obs_n = self.env.reset()
		x = tf.expand_dims(obs_n, 0)
		state = tf.repeat(x, repeats=TIME_SEQUENCE_SIZE, axis=0)

		# set up a target network
		actors_dqn_target =  [Frame_1(map_int_vect_to_int(action_space_n)+1, 11) for action_space_n in self.action_spaces]

		# for a defined maximum number of iterations
		for t in tf.range(MAX_DQN_TRAIN_ITERATIONS):

			# >>>>>> RUN one timestep where you store state, action, rw, next_state in a temporary buffer, then update to the replay buffer if able
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

			rw, estimated_trasmission_delays = instant_rw_fun(self.env, obs_n, action)
			action_init_time = self.env.clock
			obs_n, _, _, _ = self.env.step(np.array(action))


			# roll a time_step and concat observation
			next_state = roll_one_step(state, obs_n)

			# place it in the temporary experience buffer
			experience = Temporary_experience(state, action_integers, rw, next_state, action_init_time, estimated_trasmission_delays)
			temporary_experience_buffer.append(experience)

			# check out if any temporary experience is complete
			for e in temporary_experience_buffer:
				finished = e.check_update(self.env.clock, obs_n)
				# if no more updates then no need to be in the temporary experience and go to the replay buffer
				if finished:
					temporary_experience_buffer.remove(e)
					state_t, action_t, total_rw_t, next_state_t = e.value_tuple()
					instant_rw_buffer = instant_rw_buffer.write(int(e.init_time*1000), total_rw_t)
					experience_replay_buffer.push(state_t, action_t, total_rw_t, next_state_t)

			# then get ready on the next state
			state = tf.identity(next_state)

			# just some state prints
			if experience_replay_buffer.size() < REPLAY_BUFFER_SIZE and t%100 == 0:
				print("Iteration",t.numpy(),"filling replay buffer...", flush=True)
			# <<<<<<


			# >>>>>> TRAIN the network starting a specific iteration
			# only start training after the replay buffer has filled
			if experience_replay_buffer.size() == REPLAY_BUFFER_SIZE:

				# update epsilon: EXP ( (-LOG(e_start)/R^e + LOG(e_min)/R^e) * t - log(e_start) )
				self.epsilon = max( math.exp( -e_decay*(math.fmod(t,EPSILON_RENEWAL_RATE)) - math.log(e_start) ) , MIN_EPSILON)

				# 1. create a dataframe and shuffle it in batches
				train_dataset = tf.data.Dataset.from_tensor_slices(experience_replay_buffer.get_tuple())
				train_dataset = train_dataset.shuffle(buffer_size=REPLAY_BUFFER_SIZE+REPLAY_BUFFER_SIZE).batch(batch_size)

				# 2. sample just one mini batch
				for data in train_dataset:
					(train_state, train_action, train_reward, train_next_state) = data
					break

				# 3. compute targets and gradeient for every actor, then apply it with optimizer 
				for i in tf.range(self.num_actors):
					actor_train_state = train_state[:,:,i]
					actor_train_next_state = train_next_state[:,:,i]

					y_target = tf.TensorArray(dtype=tf.float32, size=batch_size)
					for b_t in tf.range(batch_size):
						x = tf.expand_dims(actor_train_next_state[b_t], 0)
						q_next_target_values = actors_dqn_target[i](x)
						t_rw_bt = tf.cast(train_reward[b_t], tf.float32)
						y_target_t = t_rw_bt + DEFAULT_GAMMA*tf.reduce_max(tf.squeeze(q_next_target_values))
						y_target = y_target.write(b_t, y_target_t)
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
					actors_dqn_target = self.update_target_network(actors_dqn_target)

				# update epsilon updating parameters every EPSILON_RENEWAL_RATE
				if t%EPSILON_RENEWAL_RATE == 0:
					e_start = e_start*EPSILON_RENEWAL_FACTOR
					e_decay = math.log(e_start)/EPSILON_RENEWAL_RATE - math.log(MIN_EPSILON)/EPSILON_RENEWAL_RATE

				# for display
				if t%10 == 0:
					print("Iteration",t.numpy()," [instant rw:", rw, "][curr epsilon:", self.epsilon,"]", flush=True) # iteration print


			# <<<<<<

		# and calculate discounted rewards
		instant_rw_buffer = instant_rw_buffer.stack().numpy()
		iter_rewards = [instant_rw_buffer[0]]
		for r in instant_rw_buffer:
			x = RW_EPS*r + (1-RW_EPS)*iter_rewards[-1]
			iter_rewards.append(x)

		# after training swap to exploit
		self.epsilon = MIN_EPSILON
		return iter_rewards
		