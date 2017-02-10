# given the price matrix, with each row corresponds to one mpn and each column corresponds to one qty, we fill up the empty elements (prices) by blind regression

import BlindRegressionV1 as BR
import pickle
import numpy as np
from bisect import bisect_left
from itertools import groupby

beta = 5
lam = 1
num_points = 10000
result_file = 'br_beta_5_lamda_1_MCU.p'

with open('Microcontrollers_data_matrix.p', 'br') as f:
	data_matrix = pickle.load(f)
f.close()

# Given the original m by n data_matrix, return the m by n estimate matrix
def blind_regression_estimate(data_matrix):
	br_obj = BR.BlindRegression(data_matrix)
	
	return br_obj.estimate_gaussian_cache_2D_all(beta, lam, num_points)

if __name__ == "__main__":
	# TODO: implement CUDA
	estimate_result = blind_regression_estimate(data_matrix)

	
	# write the result into a pickle file
	with open(result_file, 'bw') as f:
		pickle.dump(estimate_result, f)
	f.close()
