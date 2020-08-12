#  external imports
import matplotlib.pyplot as plt

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

