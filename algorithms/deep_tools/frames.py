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

# to set the tf random seed for reproducibility
def set_tf_seed(seed=1):
	tf.random.set_seed(seed)

# default output frame
class Output_Frame(tf.keras.Model):
	"""Output_Frame: just fully connected layers with multi-action output_size + value
	"""
	def __init__(self, output_sizes: List[int]):
		super(Output_Frame, self).__init__()
		# dense  layers 128, 64
		self.dense_1 = layers.Dense(128)
		self.dense_2 = layers.Dense(64)
		# output layer
		# and set up the output layers (possibly multi-discrete)
		self.output_layers = []
		self.output_values = []
		for output_size in output_sizes:
			self.output_layers.append(layers.Dense(output_size))
			self.output_values.append(layers.Dense(1))

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		""" inputs : tf.Tensor, (batch_size, time_steps, [input_shape])
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		a = self.dense_1(inputs)
		grinded = self.dense_2(a)
		return [[output_layer(grinded) for output_layer in self.output_layers],
			[output_value(grinded) for output_value in self.output_values]]
		
# a simple frame with just with dense layers
class Simple_Frame(tf.keras.Model):
	"""Simple_Frame deep neural network with dense layers
	"""
	def __init__(self, output_sizes: List[int], n_num_hidden_units: List[int] = [64, 128]):
		super(Simple_Frame, self).__init__()

		# do the number of requested hidden dense layers with RELU activation function
		self.hidden_layers = []
		for num_hidden_units in n_num_hidden_units:
			self.hidden_layers.append(layers.Dense(num_hidden_units, activation="relu"))
		
		# and set up the output layers
		self.output_f = Output_Frame(output_sizes)

	def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
		""" inputs : tf.Tensor, (batch_size, [input_shape])
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		grinded = inputs
		for hidden_layer in self.hidden_layers:
			grinded = hidden_layer(grinded)
		return self.output_f(grinded)

# CNN feature extraction then dense layers
class Conv1d_Frame(tf.keras.Model):
	""" Conv1d_Frame deep neural network with feature extration with Convulutional 1D layer
		and output dense layers
	"""
	def __init__(self, output_sizes: List[int], input_size: int):
		super(Conv1d_Frame, self).__init__()

		# 32 filters, kernal size of 3, ReLU
		self.conv1d_input = layers.Conv1D(32, 3, padding="same", activation="relu", input_shape=(None, input_size))
		# 64 filters, kernel size of 3, ReLU
		self.conv1d_hidden = layers.Conv1D(64, 3, padding="same", activation="relu")
		# connection
		self.flattener = layers.Flatten()
		# output frame
		self.output_f = Output_Frame(output_sizes)

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		""" inputs : tf.Tensor, (batch_size, time_steps, [input_shape]) : keep time_steps constant
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		a = self.conv1d_input(inputs)
		b = self.conv1d_hidden(a)
		grinded = self.flattener(b)
		return self.output_f(grinded)


class Rnn_Frame(tf.keras.Model):
	"""Rnn_Frame implements a  GRU recurrent layers followed by dense layers
	"""
	def __init__(self, output_sizes: List[int]):
		super(Rnn_Frame, self).__init__()
		
		# a GRU RNN layers
		self.rnn_input = layers.GRU(128)
		# output frame
		self.output_f = Output_Frame(output_sizes)
	
	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		""" inputs : tf.Tensor, (batch_size, time_steps, [input_shape])
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		grinded = self.rnn_input(inputs)
		return self.output_f(grinded)

class Conv1d_Rnn_Frame(tf.keras.Model):
	"""Conv1d_Rnn_Frame: conv1d layer to GRU to dense ouput
	"""
	def __init__(self, output_sizes: List[int], input_size: int):
		super(Conv1d_Rnn_Frame, self).__init__()

		# 32 filters, kernal size of 3, ReLU
		self.conv1d_input = layers.Conv1D(32, 3, padding="same", activation="relu", input_shape=(None, input_size))
		# 64 filters, kernel size of 3, ReLU
		self.conv1d_hidden = layers.Conv1D(64, 3, padding="same", activation="relu")
		# connection
		self.rnn_connector = layers.GRU(128)
		# output frame
		self.output_f = Output_Frame(output_sizes)

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		""" inputs : tf.Tensor, (batch_size, time_steps, [input_shape]) : keep time_steps constant
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		a = self.conv1d_input(inputs)
		b = self.conv1d_hidden(a)
		grinded = self.rnn_connector(b)
		return self.output_f(grinded)



