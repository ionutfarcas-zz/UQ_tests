import chaospy as cp
from numpy import zeros, random, mean, var, inf, linspace, sum
from math import erf, sqrt

class GPC:
	def get_input_samples(self, dist, size):
		samples = dist.sample(size)

		return samples

	def get_quad_points(self, quad_deg, dist):
		nodes, weights = cp.generate_quadrature(quad_deg, dist, rule="G")

		return nodes, weights

	def get_orth_poly(self, poly_deg, dist):
		P = cp.orth_ttr(poly_deg, dist, normed=True)

		return P

	def get_gpc_approx(self, P, nodes, weights, sample_func):
		gpc_approx = cp.fit_quadrature(P, nodes, weights, sample_func)

		return gpc_approx

	def get_statistics(self, gpc_approx, dist):
		E 	= cp.E(gpc_approx, dist)
		Var = cp.Var(gpc_approx, dist)

		return E, Var

	def get_gpc_h(self, dist, quad_deg, poly_deg, h):
		nodes, weights 	= self.get_quad_points(quad_deg, dist)
		P 				= self.get_orth_poly(poly_deg, dist)

		quad_size 	= len(weights)
		qoi 		= zeros(quad_size)

		for i in range(quad_size):
			f_args = nodes.T[i][0], nodes.T[i][1], nodes.T[i][2], nodes.T[i][3]
			qoi[i] = h(*f_args)

		gpc_approx = self.get_gpc_approx(P, nodes, weights, qoi)

		return gpc_approx

	def get_gpc_g(self, dist, quad_deg, poly_deg, g):
		nodes, weights 	= self.get_quad_points(quad_deg, dist)
		P 				= self.get_orth_poly(poly_deg, dist)

		quad_size 	= len(weights)
		qoi 		= zeros(quad_size)

		for i in range(quad_size):
			f_args 	= nodes.T[i][0], nodes.T[i][1], nodes.T[i][2]
			qoi[i] 	= g(*f_args)

		gpc_approx = self.get_gpc_approx(P, nodes, weights, qoi)

		return gpc_approx

	def get_gpc_f(self, dist, quad_deg, poly_deg, f, F_gaussian, get_percentile, data):
		P 				= self.get_orth_poly(poly_deg, dist)
		nodes, weights 	= self.get_quad_points(quad_deg, dist)

		quad_size = len(weights)
		quantiles = zeros(quad_size)
		func_eval = zeros(quad_size)
		
		for i in range(quad_size):
			u 				= F_gaussian(nodes.T[i][0])
			quantiles[i] 	= get_percentile(u, data)


		for i in range(quad_size):
			f_args 			= quantiles[i], nodes.T[i][1]
			func_eval[i] 	= f(*f_args)

		gpc_approx = self.get_gpc_approx(P, nodes, weights, func_eval)

		return gpc_approx

	def get_gpc_f_importance_sampling(self, dist_joint, kernel, dist_ip, quad_deg, poly_deg, f):
		P 				= self.get_orth_poly(poly_deg, dist_joint)
		nodes, weights 	= self.get_quad_points(quad_deg, dist_joint)

		dist_ratio = lambda x: kernel.pdf(x)/dist_ip.pdf(x)
		ip_weights = dist_ratio(nodes[0])

		quad_size 	= len(weights)
		func_eval 	= zeros(quad_size)
		coeff 		= zeros(poly_deg)

		for i in range(quad_size):
			f_args 			= nodes.T[i][0], nodes.T[i][1]
			func_eval[i] 	= f(*f_args)

		#coeff = [sum(func_eval*poly(*nodes)*weights*ip_weights) for poly in P]
		gpc_approx = self.get_gpc_approx(P, nodes, weights*ip_weights, func_eval)

		return gpc_approx

	def get_statistics_coeff(self, coeff):
		E 	= coeff[0]
		Var = sum([coeff[i]**2 for i in range(1, len(coeff))])

		return E, Var