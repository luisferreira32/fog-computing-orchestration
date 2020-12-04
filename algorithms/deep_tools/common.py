#!/usr/bin/env python

# external imports
import numpy as np
import copy
import tensorflow as tf
from typing import Tuple, List

# and constants imports
from algorithms.configs import DEFAULT_TRAJECTORY, DEFAULT_LEARNING_RATE


# to set the tf random seed for reproducibility
def set_tf_seed(seed=1):
	tf.random.set_seed(seed)

# near 0 number
eps = np.finfo(np.float32).eps.item()

# General Advantage Estimator ~ lbd = 1 is the TD error
def general_advantage_estimator(rewards: tf.Tensor, values: tf.Tensor, next_values: tf.Tensor, gamma: float, lbd: float = 1.0,
	standardize: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
	"""Compute the general advantage estimation per timestep
	"""
	n = tf.shape(rewards)[0]
	gaes = tf.TensorArray(dtype=tf.float32, size=n)

	# start from the end of each array
	rewards = tf.cast(rewards[::-1], dtype=tf.float32)
	values = tf.cast(values[::-1], dtype=tf.float32)
	next_values = tf.cast(next_values[::-1], dtype=tf.float32)
	delta_sum = tf.constant(0.0)
	delta_sum_shape = delta_sum.shape

	for i in tf.range(n):
		rw = rewards[i]
		v = values[i]
		nv = next_values[i]
		# A_t = delta_t + (gamma*lb) delta_t+1 + ... + (gamma*lb)^(T-t+1) delta_T-1
		# delta_t = rw_t + gamma * V_t+1 - V_t
		delta_sum = rw + gamma * nv - v + gamma * lbd * delta_sum
		delta_sum.set_shape(delta_sum_shape)
		gaes = gaes.write(i, delta_sum)
	gaes = gaes.stack()[::-1]

	returns = gaes + values
	if standardize:
		returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
		gaes = ((gaes - tf.math.reduce_mean(gaes)) / (tf.math.reduce_std(gaes) + eps))

	# advantage and expected returns Q(s,a)
	return gaes, returns

# the simple way of getting expected returns
def get_expected_returns(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
	"""Compute expected returns per timestep with just discounted sum
	G_t = sum(from t'=t to T){ gamma^(t'-t) * r_t'}
	"""

	n = tf.shape(rewards)[0]
	returns = tf.TensorArray(dtype=tf.float32, size=n)

	# Start from the end of rewards and accumulate reward sums into the returns array
	rewards = tf.cast(rewards[::-1], dtype=tf.float32)
	discounted_sum = tf.constant(0.0)
	discounted_sum_shape = discounted_sum.shape
	for i in tf.range(n):
		reward = rewards[i]
		discounted_sum = reward + gamma * discounted_sum
		discounted_sum.set_shape(discounted_sum_shape)
		returns = returns.write(i, discounted_sum)
	returns = returns.stack()[::-1]

	if standardize:
		returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

	# expected returns
	return returns

# keras loss functions
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)


def actor_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
	#print(y_true.shape, y_pred.shape)
	advantages_t = y_true
	action_probs_t = y_pred

	# actor: cross entropy with advantage as labels to scale
	actor_loss = -cce(advantages_t, action_probs_t)
	return actor_loss

def critic_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
	#print(y_true.shape, y_pred.shape)
	expected_returns = y_true
	values = y_pred
	# critic: huber loss, sum on batches
	critic_loss = huber_loss(values, expected_returns)
	return critic_loss

# --- since the keras.fit had a memory leak ---

opt = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE)

def batch_train_step(model, x_batch_train, y_batch_train, c_loss, a_loss, optimizer):
	with tf.GradientTape() as tape:
		y_pred = model(x_batch_train, training=True)
		# once again, hard coded... but can be done in a non hardcoded way!!
		value_loss =  a_loss(y_batch_train["output_1"], y_pred[0]) + a_loss(y_batch_train["output_2"], y_pred[1])+ a_loss(y_batch_train["output_3"], y_pred[2]) + a_loss(y_batch_train["output_4"], y_pred[3]) + a_loss(y_batch_train["output_5"], y_pred[4]) + a_loss(y_batch_train["output_6"], y_pred[5]) + c_loss(y_batch_train["output_7"], y_pred[6])

	grads = tape.gradient(value_loss, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))

def custom_actor_critic_fit(model, x_train, y_train, batch_size, epochs, c_loss=critic_loss, a_loss=actor_loss, optimizer=opt):
	# hardcoded - but its possible to make it flexible with an iteration over the second dimension of y_train
	output_dict = {"output_1": y_train[0],"output_2": y_train[1],"output_3": y_train[2],"output_4": y_train[3],"output_5": y_train[4],"output_6": y_train[5],"output_7": y_train[6]}
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, output_dict))
	train_dataset = train_dataset.shuffle(buffer_size=DEFAULT_TRAJECTORY*2).batch(batch_size)

	for epoch in tf.range(epochs):
		for x_batch_train, y_batch_train in train_dataset:
			batch_train_step(model, x_batch_train, y_batch_train, c_loss, a_loss, optimizer)
			
