#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# external imports
import tensorflow as tf
import numpy as np
import copy

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame, Frame_1, Frame_2
from algorithms.deep_tools.common import general_advantage_estimator, map_int_vect_to_int, map_int_to_int_vect, critic_loss, ppo_actor_loss
from algorithms.deep_tools.a2c_tools import run_actor_critic_tragectory, set_training_env_vec, training_env_vec_state

# some necesary constants
from algorithms.configs import DEFAULT_SAVE_MODELS_PATH, DEFAULT_ITERATIONS, DEFAULT_PPO_LEARNING_RATE, DEFAULT_CRITIC_LEARNING_RATE, ENV_RESET_IT
from algorithms.configs import DEFAULT_GAMMA, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_TRAJECTORY, TIME_SEQUENCE_SIZE, PARALLEL_ENVS, RW_EPS
from sim_env.configs import TIME_STEP, SIM_TIME


# the class itself
class A2c_Orchestrator(object):
	"""A2c_Orchestrator
	"""
	basic = False
	def __init__(self, env, actor_frame=Frame_1, critic_frame=Frame_2):
		super(A2c_Orchestrator, self).__init__()
		# common critic
		self.critic = critic_frame(1, 55)
		# node actors ~ each actor has two action spaces: for scheduling and for offloading
		self.actors = [actor_frame(map_int_vect_to_int(action_space_n)+1, 11) for action_space_n in env.action_space.nvec]
		self.actors_names = ["_node"+str(n.index) for n in env.nodes]
		self.num_actors = len(env.nodes)

		# meta-data
		self.name = env.case["case"]+"_rd"+str(env.rd_seed)+"_a2c_orchestrator_"+str(self.critic)
		self.env_vec = [copy.deepcopy(env) for _ in range(PARALLEL_ENVS)]
		self.action_spaces = env.action_space.nvec
		self.observation_spaces = env.observation_space.nvec

		# and aux variables
		self.act_state = None

	def __str__(self):
		return self.name

	@staticmethod
	def short_str():
		return "a2c"

	def act(self, obs_n):
		""" takes the whole system observation and returns the action step of all the nodes """
		if self.act_state is None:
			x = tf.expand_dims(obs_n, 0)
			self.act_state = tf.repeat(x, repeats=TIME_SEQUENCE_SIZE, axis=0)
		# roll one timestep and stack the new obs
		actor_state_list = tf.unstack(self.act_state)
		self.act_state = tf.stack((actor_state_list[1:]))
		self.act_state = tf.concat((self.act_state, [obs_n]), axis=0)

		# for each agent decide an action
		action = []
		for i in range(len(self.actors)):
			obs = self.act_state[:,i]
			actor = self.actors[i]
			action_space = self.action_spaces[i]
			# just one batch
			obs = tf.expand_dims(obs, 0)
			# call its model
			action_logits_t = actor(obs)
			action_i = map_int_to_int_vect(action_space, tf.random.categorical(action_logits_t,1)[0,0].numpy())
			action.append(action_i)
		return np.array(action)

	def save_models(self, saved_models_path=DEFAULT_SAVE_MODELS_PATH):
		# function to save the models of this algorithm
		complete_path = saved_models_path + self.name
		for model, name in zip(self.actors, self.actors_names):
			model.save(complete_path+name)
		self.critic.save(complete_path+"_critic")

	def load_models(self, saved_models_path=DEFAULT_SAVE_MODELS_PATH):
		# try to load the saved models
		complete_path = saved_models_path + self.name
		load_fun = tf.keras.models.load_model
		try:
			for i, name in enumerate(self.actors_names):
				self.actors[i] = load_fun(complete_path+name, compile=False)
			self.critic = load_fun(complete_path+"_critic", compile=False)
		except Exception as e:
			print("[ERROR LOG] It was not able to load the specified orchestrator model from", saved_models_path)

	def train(self, total_iterations: int = DEFAULT_ITERATIONS, trajectory_lenght: int = DEFAULT_TRAJECTORY,
		batch_size: int = DEFAULT_BATCH_SIZE, epochs: int = DEFAULT_EPOCHS, ppo_lr: int = DEFAULT_PPO_LEARNING_RATE,
		critic_lr: float = DEFAULT_CRITIC_LEARNING_RATE):

		critic_optimizer = tf.keras.optimizers.SGD(learning_rate=critic_lr)
		actor_optimizer = [tf.keras.optimizers.Adam(learning_rate=ppo_lr) for _ in range(self.num_actors)]

		# return values
		iteration_rewards = []
		# set the training env
		initial_state = set_training_env_vec(self.env_vec)
		current_state = initial_state
		# Run the model for total_iterations
		for iteration in range(total_iterations):

			# run a trajectory in each of the PARALLEL_ENVS 
			train_dataset = None # free previous dataset memory
			it_rw = 0
			for env_n in range(PARALLEL_ENVS):
				states, actions, rewards, dones, values, run_action_probs = run_actor_critic_tragectory(current_state, env_n, self, trajectory_lenght)
				advantages, target_values = general_advantage_estimator(rewards[:-1], values[:-1], values[1:], dones[1:], DEFAULT_GAMMA)
				x = tf.data.Dataset.from_tensor_slices((states[:-1], actions[:-1], run_action_probs[:-1], advantages, target_values))
				if train_dataset is None:
					train_dataset = x
				else:
					train_dataset = train_dataset.concatenate(x)
				it_rw += tf.reduce_sum(rewards).numpy()/trajectory_lenght
			
			print("Iterations",iteration," [iteration avg reward:", it_rw/PARALLEL_ENVS, "]") # iteration print

			# shuffle data and create the batches
			train_dataset = train_dataset.shuffle(buffer_size=(trajectory_lenght+batch_size)*PARALLEL_ENVS).batch(batch_size)
			#print(train_dataset.cardinality())


			# train the critic for the dataset
			critic_total_loss = 0
			for state, _, _, _, tv in train_dataset:

				# train the critic, keep batch, merge nodes
				node_state_list = tf.unstack(state, axis=1)
				joint_state = tf.concat(node_state_list, -1) # concat along the features
				with tf.GradientTape() as tape:
					values = self.critic(joint_state, training=True)
					values = tf.squeeze(values)
					loss = critic_loss(values, tv)
				grads = tape.gradient(loss, self.critic.trainable_weights)
				critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
				del tape

				critic_total_loss += loss.numpy()
				#break # only runs one minibatch - erase it to run all minibatches

			print("[CRITIC UPDATE] total loss", critic_total_loss)

			# after running trajectories train with the whole data
			for e in tf.range(epochs):
				losses = {}
				for state, action, old_action_probs, adv, _ in train_dataset:

					#and each actor
					for i in tf.range(self.num_actors):
						with tf.GradientTape() as tape:
							# 
							action_logits = self.actors[i](state[:,i], training=True)

							# for every discrete action ~ change to probs and organize it by batches
							action_probs = tf.TensorArray(dtype=tf.float32, size=batch_size)
							for t in tf.range(batch_size):
								action_probs = action_probs.write(t, tf.nn.softmax(action_logits[t])[action[t,i]])
							action_probs = action_probs.stack()

							loss = ppo_actor_loss(old_action_probs[:,i], action_probs, adv)
						grads = tape.gradient(loss, self.actors[i].trainable_weights)
						actor_optimizer[i.numpy()].apply_gradients(zip(grads, self.actors[i].trainable_weights))
						del tape

						if i.numpy() not in losses:
							losses[i.numpy()] = 0
						losses[i.numpy()] += loss.numpy()

					#break # only runs one minibatch - erase it to run all minibatches

				print("[EPOCH",e.numpy()+1,"/",epochs,"] cumulative losses:", [(x, round(y, 5)) for x, y in losses.items()]) # epoch print

			# fetch the new state: with reset given some iterations ~ in order to visit more states
			if iteration%ENV_RESET_IT == 0:
				current_state = set_training_env_vec(self.env_vec)
				#current_state = training_env_vec_state()
				#print(current_state)
			else:
				current_state = training_env_vec_state()
			# saving values
			r = it_rw/PARALLEL_ENVS
			if iteration_rewards:
				r = RW_EPS*r + (1-RW_EPS)*iteration_rewards[-1]
			iteration_rewards.append(r)

		#then return iteration rewards
		return iteration_rewards
