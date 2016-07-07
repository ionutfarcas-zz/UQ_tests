from gpc import *
from mcs import *
from kde import *
from model import *
from numpy import linspace, mean, var, arange, zeros, random, array, atleast_1d
from math import sqrt
from matplotlib.pyplot import *

if __name__ == '__main__':
	# gPC setup
	# h = f(g)
	quad_deg_h = 5
	poly_deg_h = 4
	# f
	quad_deg_f = 8
	poly_deg_f = 5
	# g
	quad_deg_g = 5
	poly_deg_g = 4
	
	# stochastic setup
	# means
	mean_x1 = 0.3
	mean_x2 = 0.2
	mean_x3 = 0.6
	mean_y2 = 0.8
	# std devs
	sdev_x1 = 0.01
	sdev_x2 = 0.01
	sdev_x3 = 0.05
	sdev_y2 = 0.04

	mean_x 		= [mean_x1, mean_x2, mean_x3]
	sdev_x 		= [sdev_x1, sdev_x2, sdev_x3]
	# no samples
	no_samples = 10000

	# gPC distributions
	dist_x1 = cp.Normal(mean_x1, sdev_x1)
	dist_x2 = cp.Normal(mean_x2, sdev_x2)
	dist_x3 = cp.Normal(mean_x3, sdev_x3)
	dist_y2 = cp.Normal(mean_y2, sdev_y2)

	dist_y1 = cp.Normal(0.7, 0.01)

	joint_dist_g 		= cp.J(dist_x1, dist_x2, dist_x3)
	joint_dist_f 		= cp.J(dist_y1, dist_y2)
	joint_dist_h 		= cp.J(dist_x1, dist_x2, dist_x3, dist_y2)
	joint_dist_f_ip 	= cp.J(dist_y1, dist_y2)

	# objects initialization
	gpc 	= GPC()
	mcs 	= MCS()
	model 	= Model()
	kde 	= KDE()
	##########################################################
	
	# chaospy based gPC
	# h = f(g) 
	h_hat 				= gpc.get_gpc_h(joint_dist_h, quad_deg_h, poly_deg_h, model.h)
	e_h_gpc, var_h_gpc 	= gpc.get_statistics(h_hat, joint_dist_h)
	samples_h 			= gpc.get_input_samples(joint_dist_h, no_samples)
	h_hat_eval 			= h_hat(*samples_h)
	kernel_h 			= kde.kde_estimation(h_hat_eval)
	x_h_gpc 			= linspace(min(h_hat_eval), max(h_hat_eval), no_samples)
	y_h_gpc 			= kernel_h(x_h_gpc)

	# g
	g_hat 				= gpc.get_gpc_g(joint_dist_g, quad_deg_g, poly_deg_g, model.g)
	samples_g 			= gpc.get_input_samples(joint_dist_g, no_samples)
	g_hat_eval 			= g_hat(*samples_g)
	
	# f
	f_hat 				= gpc.get_gpc_f(joint_dist_f, quad_deg_f, poly_deg_f, model.f, kde.F_gaussian, kde.get_percentile, g_hat_eval)
	e_f_gpc, var_f_gpc 	= gpc.get_statistics(f_hat, joint_dist_f)
	samples_f 			= gpc.get_input_samples(joint_dist_f, no_samples)
	f_hat_eval 			= f_hat(*samples_f)
	kernel_f 			= kde.kde_estimation(f_hat_eval)
	x_f_gpc 			= linspace(min(f_hat_eval), max(f_hat_eval), no_samples)
	y_f_gpc 			= kernel_f(x_f_gpc)
	##################################################################################################################################

	# chaospy based gPC + importance sampling - only for f
	kernel_g_ip 				= kde.kde_estimation(g_hat_eval, bw_method="scott")
	f_hat_ip 					= gpc.get_gpc_f_importance_sampling(joint_dist_f_ip, kernel_g_ip, dist_y1, quad_deg_f, poly_deg_f, model.f)
	e_f_gpc_ip, var_f_gpc_ip 	= gpc.get_statistics(f_hat_ip, joint_dist_f_ip)

	samples_f 					= gpc.get_input_samples(joint_dist_f_ip, no_samples)
	f_hat_ip_eval 				= f_hat_ip(*samples_f)
	kernel_f_ip 				= kde.kde_estimation(f_hat_ip_eval)
	x_f_ip_gpc 					= linspace(min(f_hat_ip_eval), max(f_hat_ip_eval), no_samples)
	y_f_ip_gpc 					= kernel_h(x_f_ip_gpc)
	##################################################################################################################################

	# MCS
	# h = f(g)
	h_samples_mcs 		= mcs.mcs_sampling_h(mean_x, sdev_x, mean_y2, sdev_y2, no_samples, model.h)
	e_h_mcs, var_h_mcs 	= mcs.get_statistics(h_samples_mcs)
	x_h_mcs 			= linspace(min(h_samples_mcs), max(h_samples_mcs), no_samples)
	y_h_mcs 			= kernel_h(x_h_mcs)

	# g
	g_samples_mcs 		= mcs.mcs_sampling_g(mean_x, sdev_x, no_samples, model.g)
	kernel_g 			= kde.kde_estimation(g_samples_mcs)
	samples_y1 			= kde.sample_est_pdf(kernel_g, no_samples)[0]


	# f
	f_samples_mcs 		= mcs.mcs_sampling_f(mean_y2, sdev_y2, no_samples, samples_y1, model.f)
	e_f_mcs, var_f_mcs 	= mcs.get_statistics(f_samples_mcs)
	x_f_mcs 			= linspace(min(f_samples_mcs), max(f_samples_mcs), no_samples)
	y_f_mcs 			= kernel_h(x_f_mcs)
	########################################################################################################

	# plotting
	plot(x_f_gpc, y_f_gpc, 'r--', label='gPC sep', linewidth=2.0)
	plot(x_f_mcs, y_f_mcs, 'g--', label='MCS sep', linewidth=2.0)
	plot(x_f_ip_gpc, y_f_ip_gpc, 'm--', label='gPC parallel', linewidth=2.0)
	plot(x_h_gpc, y_h_gpc, 'b--', label='gPC composite', linewidth=2.0)
	plot(x_h_mcs, y_h_mcs, 'k--', label='MCS composite', linewidth=2.0)
	legend(loc='best')
	grid()

	show()