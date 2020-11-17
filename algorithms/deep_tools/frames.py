#!/usr/bin/env python

# implement the networks given input and output shape (and sess?)
# simple frame : dense layers
# image frame : CNN layer + dense
# memory frame : GRU / LSTM layer + dense
# image memory frame : CNN layer + GRU / LSTM layer + dense

# essential imports for this Neural Networks frameworks
import tensorflow as tf
from tensorflow.keras import layers
# other necessary imports
from typing import Any, List, Sequence, Tuple
import tqdm
import numpy as np
import collections

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()
# to set the tf random seed for reproducibility
def set_tf_seed(seed=1):
	tf.random.set_seed(seed)

# a simple frame with just with dense layers
class Simple_Frame(tf.keras.Model):
		"""Simple_Frame deep neural network with dense layers
		"""
		def __init__(self, output_size: int, n_num_hidden_units: List[int] = [64, 128, 128, 64]):
			super(Simple_Frame, self).__init__()

			# do the number of requested hidden dense layers with RELU activation function
			self.hidden_layers = []
			for num_hidden_units in n_num_hidden_units:
				self.hidden_layers.append(layers.Dense(num_hidden_units, activation="relu"))
			# and set up the output layer
			self.output_layer = layers.Dense(output_size)

		def call(self, inputs: tf.Tensor) -> tf.Tensor:
			""" inputs : tf.Tensor, (batch_size, [input_shape])
				returns : N-D tensor (batch_size, [output_shape])
			"""
			# pass inputs on model and return the output value Tensor
			grinded = inputs
			for hidden_layer in self.hidden_layers:
				grinded = hidden_layer(grinded)
			return self.output_layer(grinded)

# CNN feature extraction then dense layers
class Conv1d_Frame(tf.keras.Model):
	""" Conv1d_Frame deep neural network with feature extration with Convulutional 1D layer
		and output dense layers
	"""
	def __init__(self, output_size: int, input_size: int):
		super(Conv1d_Frame, self).__init__()

		# 32 filters, kernal size of 3, ReLU
		self.conv1d_input = layers.Conv1D(32, 3, activation="relu", input_shape=(None, input_size))
		# 64 filters, kernel size of 3, ReLU
		self.conv1d_hidden = layers.Conv1D(64,3, activation="relu")
		# connection
		self.flattener = layers.Flatten()
		# dense  layers 128, 64
		self.dense_1 = layers.Dense(128)
		self.dense_2 = layers.Dense(64)
		# output layer
		self.ouptut_layer = layers.Dense(output_size)

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		""" inputs : tf.Tensor, (batch_size, time_steps, [input_shape]) : keep time_steps constant
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		a = self.conv1d_input(inputs)
		b = self.conv1d_hidden(a)
		c = self.flattener(b)
		d = self.dense_1(c)
		e = self.dense_2(d)
		return self.ouptut_layer(e)



