#!/usr/bin/env python

import os
import tensorflow as tf

saved_models_path = os.getcwd()+"/algorithms/saved_models/"

def save_orchestrator_models(orchestrator):
	complete_path = saved_models_path + orchestrator.name
	for model, name in zip(orchestrator.actors, orchestrator.actors_names):
		model.save(complete_path+name)
	orchestrator.critic.save(complete_path+"_critic")

load_fun = tf.keras.models.load_model
def load_orchestrator_models(orchestrator):
	complete_path = saved_models_path + orchestrator.name
	try:
		for i, name in enumerate(orchestrator.actors_names):
			orchestrator.actors[i] = load_fun(complete_path+name, compile=False)
		orchestrator.critic = load_fun(complete_path+"_critic", compile=False)
	except Exception as e:
		print("[ERROR LOG] It was not able to load the specified orchestrator model from ./algorithms/saved_models/")
		return orchestrator # without trained model fetched
	return orchestrator # with trained model
	
def fetch_orchestrator(alg, env):
	# creates the agents
	orchestrator = alg(env)
	# if there are trainned agents fetch them, otherwise, they'll be untrained
	load_orchestrator_models(orchestrator)
	return orchestrator