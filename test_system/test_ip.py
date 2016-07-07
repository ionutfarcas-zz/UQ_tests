import chaospy as cp
import numpy as np

def test_func(x):
	func = 2*x*np.exp(-x) + np.sin(x)

	return func

def get_nodes_weigths(dist, quad_deg, rule):
	nodes, weights = cp.generate_quadrature(dist, quad_deg, rule=rule)

	return nodes, weights

def standard_quad(func, nodes, weights):
	integral = np.sum(func(nodes)*weights)

	return integral

def imp_sampling_quad(func, nodes, weights, dist_ratio):
	ip_weights 	= dist_ratio(nodes)
	integral 	= np.sum(func(nodes)*weights*ip_weights)

	return integral


if __name__ == '__main__':
	quad_deg_std 	= 5
	quad_deg_ip 	= 8
	rule 			= "G"

	dist_true 	= cp.Normal(5., 0.1)
	dist_ip 	= cp.Uniform(4.5, 5.5)

	dist_ratio = lambda x: dist_true.pdf(x)/dist_ip.pdf(x)

	nodes_standard, weights_standard 	= get_nodes_weigths(quad_deg_std, dist_true, rule)
	nodes_ip, weights_ip 				= get_nodes_weigths(quad_deg_ip, dist_ip, rule)

	int_real 	= standard_quad(test_func, nodes_standard, weights_standard)
	int_ip 		= imp_sampling_quad(test_func, nodes_ip, weights_ip, dist_ratio)

	print int_real
	print int_ip