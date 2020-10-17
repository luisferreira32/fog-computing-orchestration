#  external imports
import matplotlib.pyplot as plt
import os

def graphtime(xtime=None, lines=None, xlabel="x", ylabel="y", title="default_title"):
	"""graph multiple lines in a dictionary lines with time axis xtime
	
	Parameters
	----------
	xtime=None
		the x values, in time
	d=None
		the dictionary with y values for multiple nodes
	w0=0
		number of offloaded tasks

	Return
	------
	-1 if failed, 0 otherwise
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