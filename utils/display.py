#!/usr/bin/env python

#  external imports
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import os
import numpy as np

# aux functions
def milliseconds(x, pos):
    return '%1.2f' % (x * 1e3)
# for figure configs
my_path = os.getcwd()+"/results/"
mili_formater = FuncFormatter(milliseconds)


def plt_bar(df,mili=False, title="default_title"):
	labels, data = df.keys(), df.values()
	fig, ax = plt.subplots()
	if mili: 
		ax.yaxis.set_major_formatter(mili_formater)
	# if data is arrays just do avg
	means = [np.mean(np.array(b)) if b != [] else 0 for b in data]
	plt.bar(range(len(df)), means)
	plt.xticks(range(len(df)), list(labels))
	fig.savefig(my_path+title+".png")

def plt_error_bar(df, mili=False,title="default_title"):
	labels, data = df.keys(), df.values()
	fig, ax = plt.subplots()
	if mili:
		ax.yaxis.set_major_formatter(mili_formater)
	means = [np.mean(np.array(b)) if b != [] else 0 for b in data]
	std_devs = [np.std(np.array(b)) if b != [] else 0 for b in data]
	plt.bar(range(len(df)), means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10)
	plt.xticks(range(len(df)), list(labels))
	fig.savefig(my_path+title+".png")

def plt_box_plot(df, mili=False, title="default_title"):
	labels, data = df.keys(), df.values()
	fig, ax = plt.subplots()
	if mili:
		ax.yaxis.set_major_formatter(mili_formater)
	plt.boxplot(data)
	plt.xticks(range(1, len(labels) + 1), labels)
	fig.savefig(my_path+title+".png")

def info_logs(algorithm_description, case_description, extime, average_delay, success_rate, overflow_rate):
	print("Finished",algorithm_description,"on case",case_description)
	print("extime:",extime,"s")
	print("-average delay:",average_delay,"ms")
	print("-success rate:",success_rate)
	print("-overflow rate:",overflow_rate)
