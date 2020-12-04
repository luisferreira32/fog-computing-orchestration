#!/usr/bin/env python

# external imports
import numpy as np
import copy
import tensorflow as tf
from typing import Tuple, List


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

# huber loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def actor_loss(y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> tf.Tensor:
	advantages_t = y_true
	action_probs_t = y_pred

	# actor: negative log likelihood, sum on batches
	action_log_probs_t = tf.math.log(action_probs_t)
	actor_loss = -tf.math.reduce_sum(action_log_probs_t*advantages_t) 
	return actor_loss

def critic_loss(y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> tf.Tensor:
	expected_returns = y_true
	values = y_pred
	# critic: huber loss, sum on batches
	critic_loss = huber_loss(values, expected_returns)
	return critic_loss