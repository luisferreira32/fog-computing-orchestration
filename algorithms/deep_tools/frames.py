#!/usr/bin/env python

# essential imports for this Neural Networks frameworks
import tensorflow as tf
from tensorflow.keras import layers
# other necessary imports
from algorithms.configs import TIME_SEQUENCE_SIZE


	
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
	@staticmethod
	def short_str():
		return "sf"

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		""" inputs : tf.Tensor, (batch_size, time_steps,[input_shape])
			returns : N-D tensor (batch_size, [output_shape])
		"""
		# pass inputs on model and return the output value Tensor
		x = self.dense_a(inputs[:,-1])
		x = self.dense_b(x)
		x = self.dense_c(x)
		x = self.dense_1(x)
		x = self.dense_2(x)
		return self.output_layer(x)


class Frame_1(tf.keras.Model):
	"""Frame_1 deep neural network with  specific architecture
	"""
	def __init__(self, output_size: int, feature_size: int, sequence_size: int = TIME_SEQUENCE_SIZE):
		super(Frame_1, self).__init__()

		self.conv1d_input = layers.Conv1D(32, 3, padding="same", activation="relu", input_shape=(None, sequence_size, feature_size))
		self.conv1d_hidden = layers.Conv1D(64, 3, padding="same", activation="relu")
		self.rnn_connector = layers.GRU(128)
		self.dense_1 = layers.Dense(64, activation="relu")
		self.dense_2 = layers.Dense(output_size)
		self.output_layer = layers.Dense(output_size)
		
	def __str__(self):
		return "f1"
	@staticmethod
	def short_str():
		return "f1"

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		""" inputs : tf.Tensor, (batch_size, sequence_size, features)
			returns : N-D tensor (batch_size, features)
		"""
		# pass inputs on model and return the output value Tensor
		x = self.conv1d_input(inputs)
		x = self.conv1d_hidden(x)
		x = self.rnn_connector(x)
		x = self.dense_1(x)
		x = self.dense_2(x)
		return self.output_layer(x)

class Frame_2(tf.keras.Model):
	"""Frame_2 deep neural network with  specific architecture
	"""
	def __init__(self, output_size: int, feature_size: int, sequence_size: int = TIME_SEQUENCE_SIZE):
		super(Frame_2, self).__init__()

		self.conv1d_input = layers.Conv1D(32, 3, padding="same", activation="relu", input_shape=(None, sequence_size, feature_size))
		self.conv1d_hidden = layers.Conv1D(64, 3, padding="same", activation="relu")
		self.rnn_connector = layers.GRU(128)
		self.dense_1 = layers.Dense(128, activation="relu")
		self.dense_2 = layers.Dense(64)
		self.output_layer = layers.Dense(output_size)
		
	def __str__(self):
		return "f2"
	@staticmethod
	def short_str():
		return "f2"

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		""" inputs : tf.Tensor, (batch_size, sequence_size, features)
			returns : N-D tensor (batch_size, features)
		"""
		# pass inputs on model and return the output value Tensor
		x = self.conv1d_input(inputs)
		x = self.conv1d_hidden(x)
		x = self.rnn_connector(x)
		x = self.dense_1(x)
		x = self.dense_2(x)
		return self.output_layer(x)