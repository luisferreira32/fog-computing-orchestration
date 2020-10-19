#  external imports
import matplotlib.pyplot as plt
import os

def all_graphtime(xtimes, liness, xlabels, ylabels, titles, rows=2, columns=3):
	""" To do muliple sub plots
	"""
	fig, axs = plt.subplots(rows, columns)
	r = 0; c = 0
	for i in range(len(xtimes)):
		xtime = xtimes[i]
		lines = liness[i]
		miny = 10000
		maxy = 0
		for key, line in lines.items():
			axs[r,c].plot(xtime, line, label=key)
			minya = min(line)
			maxya = max(line)
			miny = min(minya, miny)
			maxy = max(maxya, maxy)
			#axs[r,c].legend()
		handles, labels = axs[r,c].get_legend_handles_labels()

		axs[r,c].set(xlabel=xlabels[i], ylabel=ylabels[i])
		axs[r,c].set_xlim(min(xtime), max(xtime))
		axs[r,c].set_ylim(min(miny,0), maxy*1.1)
		axs[r,c].set_title(titles[i])
		if r < rows-1:
			r +=1
		elif c < columns-1:
			c+=1
			r = 0
	plt.tight_layout()
	fig.legend(handles, labels, loc='upper right')

	plt.show()


def graphtime(xtime=None, lines=None, xlabel="x", ylabel="y", title="default_title"):
	"""graph multiple lines in a dictionary lines with time axis xtime
	"""
	if lines==None or xtime==None:
		return -1

	miny = 10000
	maxy = 0
	for key, line in lines.items():
		plt.plot(xtime, line, label=key)
		minya = min(line)
		maxya = max(line)
		miny = min(minya, miny)
		maxy = max(maxya, maxy)

	plt.xlabel(xlabel)
	plt.xlim(min(xtime), max(xtime))
	plt.ylim(min(miny,0), maxy+1)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend()
	plt.show()
	return 0


def displayState(time,nodes, evq):
	# nodes state
	print("---------------------------------[%.2f]----------------------------------" % time)
	evqstring = ""
	for e in evq.q:
		evqstring += " --> " + str(e)
	print(evqstring)	
	print("-------------------------------------------------------------------------")
	for n in nodes:
		print("Node",n.name)
		# queue states
		qstring = "queue timestamps: "
		for t in n.cpuqueue:
			qstring += str(round(t.timestamp, 2)) + " "
		print(qstring)
		print("processing:",n.processing)
		# com state
		sq = "sendingq: "
		for t in n.sendq:
			sq += str(round(t[0].timestamp,2))+"->"+t[1].name+ "  "
		print(sq)
		print("transmitting:",n.transmitting)

		print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
	print("-------------------------------------------------------------------------")

	input("Press a enter to continue...")
	#os.system("clear")