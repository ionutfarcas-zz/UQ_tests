import chaospy as cp
from numpy import pi, sin, random, sum, exp
from scipy.stats import gaussian_kde

def test_function(x):
	f = x + x**2 + sin(x)*exp(-x)

	return f

def eval_integral(f, nodes, weights):
	integral = sum(f(nodes[0])*weights)

	return integral

def pdf(self, x):
	return kernel.pdf(x)

def cdf(self, x):
	return kernel.integrate_box1d(min(samples), x)

def bnd(self):
	return min(samples), max(samples)

if __name__ == '__main__':
	quad_deg = 4

	# standard approach
	dist_real 				= cp.Normal()
	nodes_std, weights_std 	= cp.generate_quadrature(quad_deg, dist_real, rule = "G")
	integral_std 			= eval_integral(test_function, nodes_std, weights_std)

	# convoluted approach
	samples 						= random.normal(0, 1, size=100000)
	kernel 							= gaussian_kde(samples)
	Dist 							= cp.construct(pdf=pdf, cdf=cdf, bnd=bnd)
	dist_approx 					= Dist()
	nodes_approx, weights_approx 	= cp.generate_quadrature(2*quad_deg, dist_approx, rule="G")
	integral_approx 				= eval_integral(test_function, nodes_approx, weights_approx)

	print integral_std
	print integral_approx