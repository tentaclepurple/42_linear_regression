import numpy as np


def ft_mse(y, y_pred):
	"""
	Calculate the mean squared error.
	"""
	mse = np.mean((y - y_pred) ** 2)
	return mse


def ft_r2_score(y, y_pred):
	"""
	Calculate the R2 score.
	"""
	ss_res = np.sum((y - y_pred) ** 2)
	ss_tot = np.sum((y - np.mean(y)) ** 2)
	r2 = 1 - ss_res / ss_tot
	return r2
