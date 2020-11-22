#!/usr/bin/env python

# external imports
import numpy as np
import copy
import tensorflow as tf
from typing import Tuple, List
# env imports
from sim_env.envrionment import Fog_env
# and runners?

# near 0 number
eps = np.finfo(np.float32).eps.item()

# General Advantage Estimator
def get_gaes(rewards: tf.Tensor, values: tf.Tensor, next_values: tf.Tensor, lbd: float,
	gamma: float, standardize: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
	"""Compute the general advantage estimation per timestep
	"""
	deltas = [r + gamma * nv - v for r, nv, v in zip(rewards, next_values, values)]
	deltas = np.stack(deltas)
	gaes = copy.deepcopy(deltas)
	for t in reversed(range(len(deltas) - 1)):
		gaes[t] = gaes[t] + gamma * lbd * gaes[t + 1]

	returns = gaes + values
	if standardize:
		gaes = (gaes - gaes.mean()) / (gaes.std() + eps)

	# advantage and expected returns Q(s,a)
	return tf.convert_to_tensor(gaes), tf.convert_to_tensor(returns)

# Or just the basic
def get_expected_returns(rewards: tf.Tensor, values: tf.Tensor,
	gamma: float, standardize: bool = True) -> tf.Tensor:
	"""Compute expected returns per timestep with just discounted sum
	G_t = sum(from t'=t to T){ gamma^(t'-t) * r_t'}
	"""

	n = tf.shape(rewards)[0]
	returns = tf.TensorArray(dtype=tf.float32, size=n)

	# Start from the end of `rewards` and accumulate reward sums
	# into the `returns` array
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
