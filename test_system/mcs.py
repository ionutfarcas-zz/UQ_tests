from numpy import zeros, random, mean, var

class MCS:
	def get_samples_x(self, mean, sdev, size):
		samples_x1 = random.normal(mean[0], sdev[0], size=size)
		samples_x2 = random.normal(mean[1], sdev[1], size=size)
		samples_x3 = random.normal(mean[2], sdev[2], size=size)

		samples    = samples_x1, samples_x2, samples_x3

		return samples

	def get_samples_y(self, mean, std_dev, size):
		samples_y2 = random.normal(mean, std_dev, size=size)

		return samples_y2
	
	def mcs_sampling_h(self, mean_x, sdev_x, mean_y, sdev_y, no_samples, h):
		samples_x = self.get_samples_x(mean_x, sdev_x, no_samples)
		samples_y = self.get_samples_y(mean_y, sdev_y, no_samples)

		h_sampled = zeros(no_samples)

		for i in range(no_samples):
			f_args 			= samples_x[0][i], samples_x[1][i], samples_x[2][i], samples_y[i]
			h_sampled[i] 	= h(*f_args)
			
		return h_sampled

	def mcs_sampling_g(self, mean_x, sdev_x, no_samples, g):
		samples_x = self.get_samples_x(mean_x, sdev_x, no_samples)

		g_sampled = zeros(no_samples)

		for i in range(no_samples):
			f_args 			= samples_x[0][i], samples_x[1][i], samples_x[2][i]
			g_sampled[i] 	= g(*f_args)
		
		return g_sampled

	def mcs_sampling_f(self, mean_y, sdev_y, no_samples, samples_y1, f):
		samples_y2 = self.get_samples_y(mean_y, sdev_y, no_samples)

		f_sampled = zeros(no_samples)

		for i in range(no_samples):
			f_args 			= samples_y1[i], samples_y2[i]
			f_sampled[i] 	= f(*f_args)

		return f_sampled

	def get_statistics(self, out_samples):
		E 	= mean(out_samples)
		Var = var(out_samples, ddof=1)

		return E, Var