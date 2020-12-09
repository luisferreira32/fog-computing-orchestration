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
	def __init__(self, output_size: int):
		super(Simple_Frame, self).__init__()

		self.dense_a = layers.Dense(64, activation="relu")
		self.dense_b = layers.Dense(128, activation="relu")
		self.dense_c = layers.Dense(256, activation="relu")
		# dense  layers 128, 64
		self.dense_1 = layers.Dense(128)
		self.dense_2 = layers.Dense(64)
		# output layers
		self.output_layer = layers.Dense(output_size)
		
	def __str__(self):
		return "sf"

	def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
		""" inputs : tf.Tensor, (batch_size, [input_shape])
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		x = self.dense_a(inputs)
		x = self.dense_b(x)
		x = self.dense_c(x)
		x = self.dense_1(x)
		x = self.dense_2(x)
		return self.output_layer(x)


class Frame_1(tf.keras.Model):
	"""Frame_1 deep neural network with  specific architecture
	"""
	def __init__(self, output_size: int):
		super(Frame_1, self).__init__()


		self.rnn_input = layers.GRU(128)
		self.dense_a = layers.Dense(128, activation="relu")
		self.dense_b = layers.Dense(256, activation="relu")
		# dense  layers 128, 64
		self.dense_1 = layers.Dense(128)
		self.dense_2 = layers.Dense(64)
		self.output_layer = layers.Dense(output_size)
		
	def __str__(self):
		return "f1"

	def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
		""" inputs : tf.Tensor, (batch_size, [input_shape])
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		x = self.rnn_input(inputs)
		x = self.dense_a(x)
		x = self.dense_b(x)
		x = self.dense_1(x)
		x = self.dense_2(x)
		return self.output_layer(x)