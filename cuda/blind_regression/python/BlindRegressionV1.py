# implement blind regression model to estimate blank entries on the matrix
import math
from heapq import nsmallest
import pickle
import sys
import datetime
import time
import numpy as np

class BlindRegression():
	def __init__(self, data_matrix):
		self._data_matrix = data_matrix
		self._dimension = self._data_matrix.shape
		# calculate "N1(u)" and "N2(i)"
		self._n1u = []
		for row in self._data_matrix:
			self._n1u.append(set([i for i in range(len(row)) if row[i] > 0]))
		self._n2i = []
		for col in self._data_matrix.T:
			self._n2i.append(set([i for i in range(len(col)) if col[i] > 0]))


	def estimate_gaussian(self, u, i, beta=2, lam=1, num_points=20):

		if beta < 2:
			print('ERROR: Invalid beta')
			return None

		# calculate "N1(u,v)"
		n1uv = []
		for v in range(self._dimension[0]):
			n1uv.append(self._n1u[u].intersection(self._n1u[v]))

		# calculate "N2(i,j)"
		n2ij = []
		for j in range(self._dimension[1]):
			n2ij.append(self._n2i[i].intersection(self._n2i[j]))

		beta_u = set([v for v in self._n2i[i] if v != u and len(n1uv[v]) >= beta])
		beta_i = set([j for j in self._n1u[u] if j != i and len(n2ij[j]) >= beta])

		beta_ui = sorted([(v, j) for v in beta_u for j in beta_i if self._data_matrix[v][j] > 0], key=lambda x:x[1])

		if len(beta_ui) == 0:
			return None

		if len(beta_ui) > num_points:
			beta_ui = nsmallest(num_points, beta_ui, key=lambda x: abs(x[1]-i))

		print(beta_ui)

		estimate = 0
		total_weight = 0
		for (v, j) in beta_ui:
			s_uv_square = self._suv_square(u, v, n1uv[v])
			s_ij_square = self._sij_square(i, j, n2ij[j])
			w_vj = math.exp(-lam*min(s_uv_square, s_ij_square))
			estimate += w_vj*(self._data_matrix[u][j] + self._data_matrix[v][i] - self._data_matrix[v][j])
			total_weight += w_vj

		return estimate / total_weight

	def estimate_gaussian_cache(self, u, i_vec, beta=2, lam=1, num_points=20):

		if beta < 2:
			print('ERROR: Invalid beta')
			return None

		estimate_result = []
		print('i_vec', i_vec)
		sys.stdout.flush()
		# calculate "N1(u,v)"
		n1uv = []
		for v in range(self._dimension[0]):
			n1uv.append(self._n1u[u].intersection(self._n1u[v]))

		s_uv_square_cache = {}

		for i in i_vec:
			print(i)
			# calculate "N2(i,j)"
			n2ij = []
			for j in range(self._dimension[1]):
				n2ij.append(self._n2i[i].intersection(self._n2i[j]))

			beta_u = set([v for v in self._n2i[i] if v != u and len(n1uv[v]) >= beta])
			beta_i = set([j for j in self._n1u[u] if j != i and len(n2ij[j]) >= beta])

			beta_ui = sorted([(v, j) for v in beta_u for j in beta_i if self._data_matrix[v][j] > 0], key=lambda x: x[1])

			if len(beta_ui) == 0:
				estimate_result.append([i, None])
				continue

			if len(beta_ui) > num_points:
				beta_ui = nsmallest(num_points, beta_ui, key=lambda x: abs(x[1] - i))

			estimate = 0
			total_weight = 0
			for (v, j) in beta_ui:
				if v in s_uv_square_cache:
					s_uv_square = s_uv_square_cache[v]
				else:
					s_uv_square = self._suv_square(u, v, n1uv[v])
					s_uv_square_cache[v] = s_uv_square

				s_ij_square = self._sij_square(i, j, n2ij[j])

				w_vj = math.exp(-lam * min(s_uv_square, s_ij_square))
				estimate += w_vj * (self._data_matrix[u][j] + self._data_matrix[v][i] - self._data_matrix[v][j])
				total_weight += w_vj

			estimate_result.append((i, estimate/total_weight))

		return estimate_result

	def estimate_gaussian_cache_2D(self, u_vec, i_vec, beta=2, lam=1, num_points=20):
		if beta < 2:
			print('ERROR: Invalid beta')
			return None

		s_uv2_cache = {}
		s_ij2_cache = {}
		estimate_result = {}

		counter = 0
		for u in u_vec:
			counter += 1
			if counter % 100 == 0:
				current_ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
				print(counter, '/', len(u_vec), 'done', current_ts)
			estimate_result[u] = []
			for i in i_vec:
				# calculate "N1(u,v)"
				n1uv = []
				for v in range(self._dimension[0]):
					n1uv.append(self._n1u[u].intersection(self._n1u[v]))
				# calculate "N2(i,j)"
				n2ij = []
				for j in range(self._dimension[1]):
					n2ij.append(self._n2i[i].intersection(self._n2i[j]))

				beta_u = set([v for v in self._n2i[i] if v != u and len(n1uv[v]) >= beta])
				beta_i = set([j for j in self._n1u[u] if j != i and len(n2ij[j]) >= beta])

				beta_ui = sorted([(v, j) for v in beta_u for j in beta_i if self._data_matrix[v][j] > 0], key=lambda x: x[1])

				if len(beta_ui) == 0:
					estimate_result[u].append([i, None])
					continue

				if len(beta_ui) > num_points:
					beta_ui = nsmallest(num_points, beta_ui, key=lambda x: abs(x[1] - i))

				estimate = 0
				total_weight = 0
				for (v, j) in beta_ui:
					if (u, v) in s_uv2_cache:
						s_uv_square = s_uv2_cache[(u, v)]
					elif (v, u) in s_uv2_cache:
						s_uv_square = s_uv2_cache[(v, u)]
					else:
						s_uv_square = self._suv_square(u, v, n1uv[v])
						s_uv2_cache[(u, v)] = s_uv_square

					if (i, j) in s_ij2_cache:
						s_ij_square = s_ij2_cache[(i, j)]
					elif (j, i) in s_ij2_cache:
						s_ij_square = s_ij2_cache[(j, i)]
					else:
						s_ij_square = self._sij_square(i, j, n2ij[j])
						s_ij2_cache[(i, j)] = s_ij_square

					w_vj = math.exp(-lam * min(s_uv_square, s_ij_square))
					estimate += w_vj * (self._data_matrix[u][j] + self._data_matrix[v][i] - self._data_matrix[v][j])
					total_weight += w_vj

				estimate_result[u].append((i, estimate / total_weight))

		return estimate_result

	def estimate_gaussian_cache_2D_all(self, beta=2, lam=1, num_points=20):
		if beta < 2:
			print('ERROR: Invalid beta')
			return None

		s_uv2_cache = {}
		s_ij2_cache = {}
		estimate_result = -1*np.ones(shape=self._dimension)

		counter = 0
		for u in range(self._dimension[0]):
			counter += 1
			if counter % 100 == 0:
				current_ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
				print(counter, '/', self._dimension[0], 'done', current_ts)
			for i in range(self._dimension[1]):
				# calculate "N1(u,v)"
				n1uv = []
				for v in range(self._dimension[0]):
					n1uv.append(self._n1u[u].intersection(self._n1u[v]))
				# calculate "N2(i,j)"
				n2ij = []
				for j in range(self._dimension[1]):
					n2ij.append(self._n2i[i].intersection(self._n2i[j]))

				beta_u = set([v for v in self._n2i[i] if v != u and len(n1uv[v]) >= beta])
				beta_i = set([j for j in self._n1u[u] if j != i and len(n2ij[j]) >= beta])

				beta_ui = sorted([(v, j) for v in beta_u for j in beta_i if self._data_matrix[v][j] > 0], key=lambda x: x[1])

				if len(beta_ui) == 0:
					continue

				if len(beta_ui) > num_points:
					beta_ui = nsmallest(num_points, beta_ui, key=lambda x: abs(x[1] - i))

				estimate = 0
				total_weight = 0
				for (v, j) in beta_ui:
					if (u, v) in s_uv2_cache:
						s_uv_square = s_uv2_cache[(u, v)]
					elif (v, u) in s_uv2_cache:
						s_uv_square = s_uv2_cache[(v, u)]
					else:
						s_uv_square = self._suv_square(u, v, n1uv[v])
						s_uv2_cache[(u, v)] = s_uv_square

					if (i, j) in s_ij2_cache:
						s_ij_square = s_ij2_cache[(i, j)]
					elif (j, i) in s_ij2_cache:
						s_ij_square = s_ij2_cache[(j, i)]
					else:
						s_ij_square = self._sij_square(i, j, n2ij[j])
						s_ij2_cache[(i, j)] = s_ij_square

					w_vj = math.exp(-lam * min(s_uv_square, s_ij_square))
					estimate += w_vj * (self._data_matrix[u][j] + self._data_matrix[v][i] - self._data_matrix[v][j])
					total_weight += w_vj

				estimate_result[u][i] = estimate / total_weight

		return estimate_result

	def _suv_square(self, u, v, uv_set):

		result = 0

		for i in uv_set:
			for j in uv_set:
				diff = (self._data_matrix[u][i] - self._data_matrix[v][i]) - (self._data_matrix[u][j] - self._data_matrix[v][j])
				result += math.pow(diff, 2)

		return result / (2*len(uv_set)*(len(uv_set)-1))

	def _sij_square(self, i, j, ij_set):

		result = 0

		for u in ij_set:
			for v in ij_set:
				diff = (self._data_matrix[u][i] - self._data_matrix[u][j]) - (self._data_matrix[v][i] - self._data_matrix[v][j])
				result += math.pow(diff, 2)

		return result / (2 * len(ij_set) * (len(ij_set) - 1))
