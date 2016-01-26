class Model:
	def c_bod(self, w, t, p):
		c_bod  = w
	 	k1, f_bod = p

		f = f_bod - k1*c_bod

		return f

	def c_do(self, w, t, p):
		c_do = w
		k1, k2, c_do_sat, c_bod = p

		f = k2*(c_do_sat - c_do) - k1*c_bod

		return f
		
	def streeter_phelps(self, w, t, p):
		c_bod, c_do = w
		k1, k2, c_do_sat, f_bod = p

		f = [f_bod - k1*c_bod,
			k2*(c_do_sat - c_do) - k1*c_bod]

		return f