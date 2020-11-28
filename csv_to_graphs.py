#!/usr/bin/env python

import os
import sys
from utils.tools import read_all_from_csvs
from utils.display import plt_bar, plt_error_bar
# just a simple file for a simple function
def csv_to_graphs():
	if len(sys.argv) < 2:
		print("[INFO] python csv_to_graphs.py [path-to-results-folder-with-csv:default-if-none-given=./results/]")
		(delays_df, success_rate_df, overflow_rate_df) = read_all_from_csvs()
		plt_bar(delays_df, mili=False, title="average_delays_1") # file should already be in mili
		plt_error_bar(success_rate_df, mili=False, title="average_success_rate_1")
		plt_error_bar(overflow_rate_df, mili=False, title="average_overflow_rate_1")
	else:
		print("[INFO] python csv_to_graphs.py [path-to-results-folder-with-csv:default-if-none-given=./results/]")
		(delays_df, success_rate_df, overflow_rate_df) = read_all_from_csvs(os.getcwd()+sys.argv[1])
		plt_bar(delays_df, mili=False, title="average_delays_1") # file should already be in mili
		plt_error_bar(success_rate_df, mili=False, title="average_success_rate_1")
		plt_error_bar(overflow_rate_df, mili=False, title="average_overflow_rate_1")

if __name__ == '__main__':
	csv_to_graphs()