from scipy.stats import gaussian_kde
from scipy.special import erf
from scipy.optimize import newton, brentq
from math import sqrt
from numpy import pi, zeros, mean, log, exp, inf
import statsmodels.api as sm

class KDE:
	def kde_estimation(self, gpc_approx):
		kernel = gaussian_kde(gpc_approx, bw_method="scott")

		return kernel

	def get_h(self, kernel):
		h = kernel.factor

		return h

	def sample_est_pdf(self, size, kernel):
		return kernel.resample(size)

	def F_gaussian(self, x):
		return 0.5*(1. + erf(x/sqrt(2.)))

	def F_kde(self, m, h, data, x):
		int_sum = [float(self.F_gaussian((x - x_i)/h)) for x_i in data]
		int_sum = sum(int_sum)
		
		F_x = int_sum/m

		return F_x

	def ecdf(self, data, x):
		cdf = sm.distributions.ECDF(data)
		F_x = cdf(x)

		return F_x

	def get_quantile_brent(self, a, b, xtol, rtol, maxiter, p, data):
		f = lambda x: self.ecdf(data, x) - p
		quantile = brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter)
		
		return quantile