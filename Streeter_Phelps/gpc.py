import chaospy as cp
from numpy import zeros, random, mean, var, inf
from math import erf, sqrt

class GPC:
	def F_gaussian(self, x):
		return 0.5*(1. + erf(x/sqrt(2.)))

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
		E = cp.E(gpc_approx, dist)
		Var = cp.Var(gpc_approx, dist)

		return E, Var

	def get_gpc_h(self, ode_sys, solver, dist, quad_deg, poly_deg, init_cond, t, pos_qoi, k1, k2):
		nodes, weights = self.get_quad_points(quad_deg, dist)
		P = self.get_orth_poly(poly_deg, dist)

		quad_size = len(weights)
		qoi = zeros((quad_size, 2))

		for i in range(quad_size):
			f_args = k1, k2, nodes.T[i][0], nodes.T[i][1]
			# sol = solver.ode_stiff_solve(ode_sys, init_cond, t, f_args)
			sol = solver.ode_solve(ode_sys, init_cond, t, f_args)
			qoi[i] = sol[pos_qoi]

		gpc_approx = self.get_gpc_approx(P, nodes, weights, qoi)

		return gpc_approx

	def get_gpc_g(self, ode_sys, solver, dist, quad_deg, poly_deg, init_cond, t, pos_qoi, k1):
		nodes, weights = self.get_quad_points(quad_deg, dist)
		P = self.get_orth_poly(poly_deg, dist)

		quad_size = len(weights)
		qoi = zeros(quad_size)

		for i in range(quad_size):
			f_args = k1, nodes.T[i]
			# sol = solver.ode_stiff_solve(ode_sys, init_cond, t, f_args)
			sol = solver.ode_solve(ode_sys, init_cond, t, f_args)
			qoi[i] = sol[pos_qoi]

		gpc_approx = self.get_gpc_approx(P, nodes, weights, qoi)

		return gpc_approx

	def get_gpc_f(self, func, kde, solver, dist, poly_deg, quad_deg, maxiter, xtol, rtol, init_cond, t, pos_qoi, data, k1, k2):
		P = self.get_orth_poly(poly_deg, dist)
		nodes, weights = self.get_quad_points(quad_deg, dist)

		quad_size = len(weights)
		quantiles = zeros(quad_size)

		a = -10.
		b = 10.
		for i in range(quad_size):
			u = self.F_gaussian(nodes.T[i][1])
			quantiles[i] = kde.get_quantile_brent(a, b, xtol, rtol, maxiter, u, data)

		func_eval = zeros(quad_size)

		for i in range(quad_size):
			f_args = k1, k2, nodes.T[i][0], quantiles[i]
			# sol = solver.ode_stiff_solve(func, init_cond, t, f_args)
			sol = solver.ode_solve(func, init_cond, t, f_args)
			func_eval[i] = sol[pos_qoi]

		gpc_approx = self.get_gpc_approx(P, nodes, weights, func_eval)

		return gpc_approx

	def get_input_samples(self, dist, size):
		samples = dist.sample(size=size)

		return samples