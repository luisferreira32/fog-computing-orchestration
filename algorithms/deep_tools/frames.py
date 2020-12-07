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
import numpy as np
import collections


	
# a simple frame with just with dense layers
class Simple_Frame(tf.keras.Model):
	"""Simple_Frame deep neural network with dense layers
	"""
	def __init__(self, output_size: int, n_num_hidden_units: List[int] = [64, 128]):
		super(Simple_Frame, self).__init__()

		# do the number of requested hidden dense layers with RELU activation function
		self.hidden_layers = []
		for num_hidden_units in n_num_hidden_units:
			self.hidden_layers.append(layers.Dense(num_hidden_units, activation="relu"))
		# dense  layers 128, 64
		self.dense_1 = layers.Dense(128)
		self.dense_2 = layers.Dense(64)
		# output layers
		self.output_layer = layers.Dense(output_size)
		for output_size in output_sizes:
			self.output_layers.append() # it's gonna return logits without activation="softmax"
		
	def __str__(self):
		return "sf"

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		""" inputs : tf.Tensor, (batch_size, [input_shape])
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		x = inputs
		for hidden_layer in self.hidden_layers:
			x = hidden_layer(x)
		x = self.dense_1(x)
		x = self.dense_2(x)
		return self.output_layer(x)



