#!/usr/bin/env python

import os
import sys
from utils.tools import read_all_from_csvs
from utils.display import plt_bar, plt_error_bar
# just a simple file for a simple function
def csv_to_graphs():
	print("[INFO] python csv_to_graphs.py [path-to-results-folder-with-csv:default=./results/] [case:default=n1 n2 n3 h1 h2 h3]")

	if len(sys.argv) < 2:
		(delays_df, success_rate_df, overflow_rate_df) = read_all_from_csvs()
	elif len(sys.argv) < 3:
		(delays_df, success_rate_df, overflow_rate_df) = read_all_from_csvs(os.getcwd()+sys.argv[1])
	else:
		cases = [sys.argv[i] for i in range(2,len(sys.argv))]
		(delays_df, success_rate_df, overflow_rate_df) = read_all_from_csvs(os.getcwd()+sys.argv[1], cases)

	plt_bar(delays_df, mili=False, title="average_delays_"+"_".join(cases)) # csv file should already be in mili
	plt_error_bar(success_rate_df, mili=False, title="average_success_rate_"+"_".join(cases))
	plt_error_bar(overflow_rate_df, mili=False, title="average_overflow_rate_"+"_".join(cases))

if __name__ == '__main__':
	csv_to_graphs()