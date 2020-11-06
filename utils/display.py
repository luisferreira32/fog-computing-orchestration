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
my_path = os.getcwd()
mili_formater = FuncFormatter(milliseconds)


def plt_bar(x, bars,mili=False, title="default_title"):
	fig, ax = plt.subplots()
	if mili: 
		ax.yaxis.set_major_formatter(mili_formater)
	plt.bar(x,bars)
	plt.xticks(x, tuple(x))
	fig.savefig(my_path+"/results/"+title+".png")

def plt_error_bar(x, bars_info, mili=False,title="default_title"):
	fig, ax = plt.subplots()
	if mili:
		ax.yaxis.set_major_formatter(mili_formater)
	means = [np.mean(np.array(b)) if b != [] else 0 for b in bars_info]
	std_devs = [np.std(np.array(b)) if b != [] else 0 for b in bars_info]
	plt.bar(x, means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10)
	plt.xticks(x, tuple(x))
	fig.savefig(my_path+"/results/"+title+".png")