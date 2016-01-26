from scipy.integrate import odeint
from numpy import floor
from odespy import Vode, Lsode, RK4, RKFehlberg, DormandPrince

class Solver:
	def __init__(self, abserr, relerr):
		self.abserr = abserr
		self.relerr = relerr

	def ode_solve(self, f, init_cond, t, f_args):
		sol = odeint(f, init_cond, t, args=(f_args,))

		return sol

	def ode_stiff_solve(self, f, init_cond, t, *f_args):
		solver = DormandPrince(f, rtol=self.relerr, atol=self.abserr,
			adams_or_bdf='bdf', f_args=f_args)
		# solver = RKFehlberg(f, rtol=self.relerr, atol=self.abserr,
		# 	f_args=f_args)
		solver.set_initial_condition(init_cond)

		sol, time = solver.solve(t)

		return sol