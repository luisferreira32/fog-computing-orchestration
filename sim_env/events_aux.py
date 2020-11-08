#!/usr/bin/env python

def is_arrival_on_slice(ev, node, k):
	if ev.classtype == "Task_arrival" and ev.node == node and ev.k == k:
		return True
	return False

def is_offload_arrival_event(ev, time):
	if ev.classtype == "Task_arrival" and ev.task_time() < clock:
		return True
	return False