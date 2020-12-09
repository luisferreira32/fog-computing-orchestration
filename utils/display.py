#!/usr/bin/env python

#  external imports
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import os
import numpy as np

# --- graphical display ---
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
	plt.tight_layout()
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
	plt.tight_layout()
	fig.savefig(my_path+title+".png")

def plt_box_plot(df, mili=False, title="default_title"):
	labels, data = df.keys(), df.values()
	fig, ax = plt.subplots()
	if mili:
		ax.yaxis.set_major_formatter(mili_formater)
	plt.boxplot(data)
	plt.xticks(range(1, len(labels) + 1), labels)
	plt.tight_layout()
	fig.savefig(my_path+title+".png")

def plt_line_plot(df, normalize=False, title="default_line_plt"):
	for key, values in df.items():
		if normalize:
			values = [v/(max(values)-min(values)) for v in values]
		plt.plot(values, label=key)
	plt.savefig(my_path+title+".png")

# --- text log display ---
# aux function
def info_gather(compiled_info, info_n):
	#print(compiled_info)
	for key, info in info_n.items():
		compiled_info["delay_sum"] += sum(info["delay_list"])
		compiled_info["succeeded"] += len(info["delay_list"])
		compiled_info["overflowed"] += info["overflow"]
		compiled_info["discarded"] += info["discarded"]
	compiled_info["total"] = compiled_info["discarded"] + compiled_info["succeeded"] + compiled_info["overflowed"]
	if compiled_info["total"] > 0:
		compiled_info["average_delay"] = compiled_info["delay_sum"]/compiled_info["total"]
		compiled_info["success_rate"] = compiled_info["succeeded"]/compiled_info["total"]
		compiled_info["overflow_rate"] = compiled_info["overflowed"]/compiled_info["total"]
	return compiled_info

def info_gather_init():
	return {"delay_sum":0, "succeeded":0, "overflowed":0, "discarded":0}

def info_logs(key, extime, compiled_info):
	print("[INFO LOG] Finished",key,"in",extime,"s")
	print("[1/4] total tasks:",compiled_info["total"])
	print("[2/4] average delay:",round(1000*compiled_info["average_delay"],5),"ms")
	print("[3/4] success rate:",round(compiled_info["success_rate"],5))
	print("[4/4] overflow rate:",round(compiled_info["overflow_rate"],5))
