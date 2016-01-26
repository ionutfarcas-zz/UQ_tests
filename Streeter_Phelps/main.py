from gpc import *
from kde import *
from model import *
from solver import *
from plotter import *
from numpy import linspace, mean, var, arange, zeros, random
from math import sqrt

if __name__ == '__main__':
	atol = 1e-5
	rtol = 1e-3
	maxiter = 100
	quad_deg_h = 8
	poly_deg_h = 3
	quad_deg_g = 8
	poly_deg_g = 3
	quad_deg_f = 8
	poly_deg_f = 5
	t_end = 25.
	time_steps = 1000
	t = linspace(0, t_end, time_steps)

	k1 = 0.3
	k2 = 0.4
	c_bod_in = 7.33
	c_do_in = 8.5

	f_bod_mean = 1.0
	f_bod_stddev = 0.1
	c_do_sat_mean = 11.
	c_do_sat_stddev = 1.1

	init_cond_h = [c_bod_in, c_do_in]
	init_cond_g = c_bod_in
	init_cond_f = c_do_in

	t_qoi = [len(t)/10, len(t)/2, len(t) - len(t)/3, len(t) - 1]
	samples_size = 50000

	gpc = GPC()
	solver = Solver(atol, rtol)
	model = Model()
	kde = KDE()
	plotter = Plotter()

	dist_f_bod = cp.Normal(f_bod_mean, f_bod_stddev)
	dist_c_do_sat = cp.Normal(c_do_sat_mean, c_do_sat_stddev)
	joint_dist_h = cp.J(dist_c_do_sat, dist_f_bod)

	dist_f_bod_standard = cp.Normal()
	joint_dist_f = cp.J(dist_c_do_sat, dist_f_bod_standard)

	exp_bb = []
	var_bb = []
	exp_mchange = []
	var_mchange = []

	no_sims = len(t_qoi)
	t_qoi_label = []

	for t_q in t_qoi:
		h_hat = gpc.get_gpc_h(model.streeter_phelps, solver, joint_dist_h, quad_deg_h, poly_deg_h, init_cond_h, t, t_q, k1, k2)
		e_h, var_h = gpc.get_statistics(h_hat, joint_dist_h)

		g_hat = gpc.get_gpc_g(model.c_bod, solver, dist_f_bod, quad_deg_g, poly_deg_g, init_cond_g, t, t_q, k1)
		samples_g = gpc.get_input_samples(dist_f_bod, samples_size)

		g_hat_eval = g_hat(samples_g)
		g_hat_eval = sorted(g_hat_eval)

		f_hat = gpc.get_gpc_f(model.c_do, kde, solver, joint_dist_f, poly_deg_f, quad_deg_f, maxiter, atol, rtol, init_cond_f, t, t_q, g_hat_eval, k1, k2)
		e_f, var_f = gpc.get_statistics(f_hat, joint_dist_f)	

		exp_bb.append(e_h[1])
		var_bb.append(var_h[1])	

		exp_mchange.append(e_f)
		var_mchange.append(var_f)

		t_qoi_label.append(t_q)	

	index = arange(no_sims)
	t_qoi_label = tuple(t_qoi_label)
	plotter.bar_chart(index, exp_bb, exp_mchange, var_bb, var_mchange, 'expectation', 'variance', 't_qoi', t_qoi_label)