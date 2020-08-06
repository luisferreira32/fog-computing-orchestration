#  external imports
import matplotlib.pyplot as plt

def graphtime(xtime=None, lines=None, xlabel="x", ylabel="y"):
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

	for key, line in lines.items():
		plt.plot(xtime, line, label=key)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel) 
	plt.show()
	return 0


def appendict(lines=None, key=None, value=None):
	"""appends information to a dictionary called lines
	
	Parameters
	----------
	lines=None
		the dictionary with information
	key=None
		the key of the dictionary
	value=None
		the value to append

	Return
	------
	-1 if failed, 0 otherwise
	"""
	if lines == None or key == None or value == None:
		return -1

	if key in lines:
		lines[key].append(value)
	else:
		lines[key] = [value]

	return 0