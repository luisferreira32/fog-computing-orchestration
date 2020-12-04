#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# and external imports
import tensorflow as tf

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame
from algorithms.deep_tools.common import general_advantage_estimator, actor_loss, critic_loss

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE, DEFAULT_ACTION_SPACE
from sim_env.configs import N_NODES, DEFAULT_SLICES, TOTAL_TIME_STEPS

# optimizer to apply the gradient change
opt = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE)

# the class itself
class A2C_Agent(object):
	"""A2C_Agent
	"""
	basic = False
	def __init__(self, n, action_space=DEFAULT_ACTION_SPACE, model_frame=Simple_Frame):
		super(A2C_Agent, self).__init__()
		# actual agent - the NN
		self.model = model_frame(action_space)		
		# meta-data
		self.name = "node_"+str(n)+"_a2c_agent"
		self.action_space = action_space
		self.gamma = 0.99

	def __str__(self):
		return self.name

	@staticmethod
	def short_str():
		return "a2c"

	def act(self, obs, batches=1):
		# wrapp in batches
		if batches == 1:
			obs = tf.expand_dims(obs, 0)
		# call its model
		model_output = self.model(obs)
		action_logits_t = model_output[:-1]
		# Since it's multi-discrete, for every discrete set of actions:
		action_i = []
		for action_logits_t_k in action_logits_t:
			# Sample next action from the action probability distribution
			action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
			action_i.append(action_i_k)
		# return the action for this agent
		return action_i

	def train(self, states, action_probs, actions, values, rw, dones, batch_size, epochs):
		# compile with chosen loss (there are N outputs but will have a combined loss calculation!) and optimizer
		losses = [actor_loss for _ in range(len(self.action_space))]
		losses.append(critic_loss)
		self.model.compile(optimizer=opt, loss=losses)

		# for each time step set up policy_targets (N sized action space), and value_target (1 critic)
		# the critic target is something like expected_returns
		# the actor target is the advantage on the action[] taken, else is zero: policy_targets = [zeros[time_steps, discrete_actions] for _ in outputlayers]
		# policy_targets[i][k][action] = advantage[i]
		# inputs: rw, values, next values, gamma, lambda
		advantage, value_target = general_advantage_estimator(rw[:-1], values[:-1], values[1:], self.gamma)
		n = tf.shape(action_probs[:-1])[0]
		policy_targets = []
		for a, num_actions in enumerate(self.action_space):
			policy_target = tf.TensorArray(dtype=tf.float32, size=n)
			for t in tf.range(n):
				advantage_target = tf.TensorArray(dtype=tf.float32, size=num_actions) # possible actions include num_actions
				advantage_target = advantage_target.write(actions[t][a], advantage[t])
				policy_target = policy_target.write(t, advantage_target.stack())
			policy_target = policy_target.stack()
			policy_targets.append(policy_target)
		y_true = policy_targets
		y_true.append(value_target)
		print([y.shape for y in y_true])
		print([self.action_space])
		# we actually can't use the last timestep since we're using TD methods
		self.model.fit(states[:-1], y_true, batch_size=batch_size, epochs=epochs)

