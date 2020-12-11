#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# and external imports
import tensorflow as tf
import numpy as np

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame, Frame_1
from algorithms.deep_tools.common import general_advantage_estimator, map_int_vect_to_int, map_int_to_int_vect, critic_loss, ppo_actor_loss
from algorithms.deep_tools.trainners import run_actor_critic_tragectory, set_training_env

# some necesary constants
from algorithms.configs import DEFAULT_SAVE_MODELS_PATH, DEFAULT_ITERATIONS, DEFAULT_PPO_LEARNING_RATE, DEFAULT_CRITIC_LEARNING_RATE
from algorithms.configs import DEFAULT_GAMMA, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_TRAJECTORY
from sim_env.configs import TIME_STEP, SIM_TIME


# the class itself
class A2c_Orchestrator(object):
	"""A2c_Orchestrator
	"""
	basic = False
	def __init__(self, env, actor_frame=Frame_1, critic_frame=Frame_1):
		super(A2c_Orchestrator, self).__init__()
		# common critic
		self.critic = critic_frame(1)
		# node actors ~ each actor has two action spaces: for scheduling and for offloading
		self.actors = [actor_frame(map_int_vect_to_int(action_space_n)+1) for action_space_n in env.action_space.nvec]
		self.actors_names = ["_node"+str(n.index) for n in env.nodes]
		self.num_actors = len(env.nodes)

		# meta-data
		self.name = env.case["case"]+"_rd"+str(env.rd_seed)+"_a2c_orchestrator_"+str(self.critic)
		self.env = env
		self.action_spaces = env.action_space.nvec
		self.observation_spaces = env.observation_space.nvec

	def __str__(self):
		return self.name

	@staticmethod
	def short_str():
		return "a2c"

	def act(self, obs_n):
		""" takes the whole system observation and returns the action step of all the nodes """

		# for each agent decide an action
		action = []
		for obs, actor, action_space in zip(obs_n, self.actors, self.action_spaces):
			obs = tf.expand_dims(obs, 0)
			# call its model
			action_logits_t = actor(obs)
			# Since it's multi-discrete, for every discrete set of actions:
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
		critic_lr: float = DEFAULT_CRITIC_LEARNING_RATE, save: bool = False):

		critic_optimizer = tf.keras.optimizers.SGD(learning_rate=critic_lr)
		actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ppo_lr)

		# return values
		iteration_rewards = []
		# set the training env
		initial_state = set_training_env(self.env)
		current_state = initial_state
		# Run the model for total_iterations
		for iteration in range(total_iterations):
			states, actions, rewards, dones, values, run_action_probs = run_actor_critic_tragectory(current_state, self, trajectory_lenght)
			print("Iterations",iteration," [iteration avg reward:", tf.reduce_sum(rewards).numpy()/trajectory_lenght, "]") # iteration print

			train_dataset = tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, values, run_action_probs))
			train_dataset = train_dataset.shuffle(buffer_size=trajectory_lenght+batch_size).batch(batch_size)

			for e in tf.range(epochs):
				losses = {"critic": 0, 0:0, 1:0, 2:0, 3:0, 4:0}
				for state, action, rw, done, v, old_action_probs in train_dataset:
					advantages, target_values = general_advantage_estimator(rw[:-1], v[:-1], v[1:], done[1:], DEFAULT_GAMMA)

					# train the critic
					joint_state = tf.reshape(state,  [tf.shape(state)[0], tf.shape(state)[1]*tf.shape(state)[2]]) # keep batch, merge nodes
					with tf.GradientTape() as tape:
						values = self.critic(joint_state, training=True)
						values = tf.squeeze(values)
						loss = critic_loss(values[:-1], target_values)
					grads = tape.gradient(loss, self.critic.trainable_weights)
					critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
					del tape

					losses["critic"] += loss.numpy()

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

							loss = ppo_actor_loss(old_action_probs[:-1,i], action_probs[:-1], advantages)
						grads = tape.gradient(loss, self.actors[i].trainable_weights)
						actor_optimizer.apply_gradients(zip(grads, self.actors[i].trainable_weights))
						del tape

						losses[i.numpy()] += loss.numpy()

				print("[EPOCH",e.numpy()+1,"/",epochs,"] cumulative losses:", losses) # epoch print

			# reset the env if needed
			if self.env.clock + trajectory_lenght*TIME_STEP >= SIM_TIME:
				current_state = self.env.reset()
			current_state = self.env._get_state_obs()
			# saving values
			iteration_rewards.append(tf.reduce_sum(rewards).numpy()/trajectory_lenght)

		# save trained orchestrator, then return it
		if save:
			orchestrator.save_models()
		return iteration_rewards
