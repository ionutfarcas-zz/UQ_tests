from numpy import exp, log, sin, cos, sqrt

class Model:
	def g(self, x1, x2, x3):
	 	func = 0.2 + exp(-x1) + x2**2/3. - 0.2*x3

		return func

	def f(self, y1, y2):
		func = y1 + 0.3*y2**2 - 0.1

		return func
		
	def h(self, x1, x2, x3, y2):
		func = self.f(self.g(x1, x2, x3), y2)

		return func