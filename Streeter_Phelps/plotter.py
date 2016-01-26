import matplotlib.pyplot as plt

class Plotter:
	def bar_chart(self, index, expectation_bb, expectation_mchange, variance_bb, variance_mchange, ylabel_1, ylabel_2, xlabel, t_qoi):
		fig, ax = plt.subplots(2, 1)
		width = 0.05

		mean_cs = ax[0].bar(index, expectation_bb, width, color='r', yerr=None)
		mean_kde = ax[0].bar(index + width, expectation_mchange, width, color='b', yerr=None)

		var_cs = ax[1].bar(index, variance_bb, width, color='r', yerr=None)
		var_kde = ax[1].bar(index + width, variance_mchange, width, color='b', yerr=None)

		ax[0].set_ylabel(ylabel_1)
		ax[0].set_xlabel(xlabel)
		ax[0].legend((mean_cs[0], mean_kde[0]), ('black box', 'measure change'), loc = "center", fontsize=15, bbox_to_anchor=(0.5, 1.1), ncol = 2)
		ax[0].set_xticks(index + width)
		ax[0].set_xticklabels(t_qoi)

		ax[1].set_ylabel(ylabel_2)
		ax[1].set_xlabel(xlabel)
		ax[1].set_xticks(index + width)
		ax[1].set_xticklabels(t_qoi)

		plt.show()