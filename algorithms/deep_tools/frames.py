#!/usr/bin/env python

# implement the networks given input and output shape (and sess?)
# simple frame : dense layers
# image frame : CNN layer + dense
# memory frame : GRU / LSTM layer + dense
# image memory frame : CNN layer + GRU / LSTM layer + dense

# essential imports for this Neural Networks frameworks
import tensorflow as tf 

def set_tf_seed(seed=1):
	tf.random.set_seed(seed)