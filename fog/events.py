# do a deque of events to be sorted

class Event(object):
	"""docstring for Event"""
	def __init__(self, start, end):
		self.start = start
		self.end = end


class Recieving(Event):
	"""docstring for Recieving"""
	def __init__(self, start, end):
		super(Recieving, self).__init__(start, end)


class Sending(Event):
	"""docstring for Sending"""
	def __init__(self, start, end):
		super(Sending, self).__init__(start, end)


class Processing(Event):
	"""docstring for Processing"""
	def __init__(self, start, end):
		super(Processing, self).__init__(start, end)

		
		
		