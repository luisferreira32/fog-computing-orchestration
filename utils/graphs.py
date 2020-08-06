#  external imports
import matplotlib.pyplot as plt

def graphqueues(xtime, queues):
	x=1
	for queue in queues:
		plt.plot(xtime, queue, label="node "+str(x))
		x+=1
	plt.xlabel('x - time')
	plt.ylabel('y - queue size') 
	plt.show()