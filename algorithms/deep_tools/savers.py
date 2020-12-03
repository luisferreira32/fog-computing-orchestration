#!/usr/bin/env python

import os
import tensorflow as tf

saved_models_path = os.getcwd()+"/algorithms/saved_models/"

def save_agent_models(agent):
	arch_path = saved_models_path + str(agent.model) + "/"
	agent.model.save(arch_path+agent.name)

def load_agent_models(agent):
	arch_path = saved_models_path + str(agent.model) + "/"
	try:
		agent.model = tf.keras.models.load_model(arch_path+agent.name, compile=False)
	except Exception as e:
		print("[ERROR LOG] It was not able to load the specified agent model from ./algorithsm/saved_models/")
		return agent # without trained model fetched
	return agent # with trained model
	
def fetch_agents(env, alg):
	# creates the agents
	agents = [alg(n.index, env.action_space.nvec[i]) for i,n in enumerate(env.nodes)]
	# if there are trainned agents fetch them, otherwise, they'll be untrained
	for agent in agents:
		agent = load_agent_models(agent)
	return agents