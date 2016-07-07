from scipy.stats import gaussian_kde
from scipy.special import erf
from scipy.optimize import newton, brentq, brenth, ridder, bisect
from math import sqrt
from numpy import pi, zeros, mean, log, exp, sum, percentile
from statsmodels.api import distributions

class KDE:
	def kde_estimation(self, gpc_approx, bw_method="scott"):
		kernel = gaussian_kde(gpc_approx, bw_method=bw_method)

		return kernel

	def get_h(self, kernel):
		h = kernel.factor

		return h

	def Gaussian_kernel(self, x):
		kernel = 1/sqrt(2*pi)*exp(-x**2/2.0)

		return kernel

	def Gaussian_kde(self, data, h, x):
		n = len(data)
		kde_est = 1.0/(n*h)*sum([self.Gaussian_kernel((x - xi)/h) for xi in data])

		return kde_est

	def sample_est_pdf(self, kernel, size):
		return kernel.resample(size)

	def F_gaussian(self, x):
		return 0.5*(1. + erf(x/sqrt(2.)))

	def ecdf(self, data, x):
		cdf = distributions.ECDF(data)
		F_x = cdf(x)

		return F_x

	def get_quantile_brent(self, a, b, xtol, rtol, maxiter, p, data):
		f = lambda x: self.ecdf(data, x) - p

		quantile = brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter)
		
		return quantile

	def get_percentile(self, p, data):
		quantile = percentile(data, p*100, interpolation='linear')

		return quantile