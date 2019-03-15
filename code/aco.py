#!/usr/bin/env python3

import random
import numpy as np

class Graph(object):

	def __init__(self, matrix, n, m, initial_pheromone):

		self.n = n  # number of vertices
		self.m = m  # number of edges
		self.matrix = matrix # edge weights
		
		# pheromone matrix
		self.pheromone = [[0 if i == j else initial_pheromone for j in range(n)]
						  for i in range(n)]

		# desirability function matrix - heuristic information
		self.eta = [[0 if i == j else matrix[i][j] for j in range(n)]
					for i in range(n)]


class ACO(object):

	def __init__(self, ants, iterations, alpha, beta, evaporation, q,
				 strategy, update_pheromone, min_ph, max_ph):
		"""
        	ants: number of ants
        	iterations: number of iterations
        	alpha: relative importance of pheromone
        	beta: relative importance of edge weights
        	evaporation: pheromone evaporation rate 
        	q: pheromone intensity
        	strategy: pheromone update strategy
        	update_pheromone: pheromone update strategy
        	min_ph: inferior limit - strategy max-min
        	max_ph: upper limit - strategy max-min
        """
		self.Q = q
		self.evaporation = evaporation
		self.beta = beta
		self.alpha = alpha
		self.ants = ants
		self.iterations = iterations
		self.update_stategy = strategy
		self.update_pheromone = update_pheromone
		self.min_ph = min_ph
		self.max_ph = max_ph



	def pheromone_evaporation(self, graph):
		for i, row in enumerate(graph.pheromone):
			for j, col in enumerate(row):
				graph.pheromone[i][j] *= (1 - self.evaporation)


	def update_pheromone(self, graph, ants):
		for ant in ants:
			if ant.valid_path:
				path = ant.path
				for k in range(1, len(path)):
					i = path[k - 1]
					j = path[k]
					if self.update_stategy == 1: # ant-density system
						pheromone_delta = self.Q*(1 - 1/graph.matrix[i][j])
					else: # ant-cycle system
						pheromone_delta = self.Q*(1 - 1/ant.total_cost)
					graph.pheromone[i][j] += pheromone_delta


	def elitist(self, graph, best_solution):
		for k in range(1, len(best_solution)):
			i = best_solution[k - 1]
			j = best_solution[k]
			graph.pheromone[i][j] += 5


	def max_min(self, graph, ant):

		for i, row in enumerate(graph.pheromone):
			for j, col in enumerate(row):
				graph.pheromone[i][j] *= (1 - self.evaporation)
				if graph.pheromone[i][j] > self.max_ph:
					graph.pheromone[i][j] = self.max_ph
				elif graph.pheromone[i][j] < self.min_ph:
					graph.pheromone[i][j] = self.min_ph

		path = ant.path
		for k in range(1, len(path)):
			i = path[k - 1]
			j = path[k]
			if self.update_stategy == 1:
				pheromone_delta = self.Q*(1 - 1/graph.matrix[i][j])
			else:
				pheromone_delta = self.Q*(1 - 1/ant.total_cost)
			graph.pheromone[i][j] += pheromone_delta
			if graph.pheromone[i][j] > self.max_ph:
				graph.pheromone[i][j] = self.max_ph
			elif graph.pheromone[i][j] < self.min_ph:
				graph.pheromone[i][j] = self.min_ph


	def build_solutions(self, graph, ants):

		for ant in ants:
			for i in range(graph.n - 1):
				k = ant.select_next()
				if k == -1:		# invalid edge
					ant.valid_path = False
					ant.total_cost = -1
					break
				elif k == graph.n - 1:
					break


	def solve(self, graph, repetition):

		best_cost = 0
		best_solution = []
		ant_global = None

		worst_cost = 99999

		stats = []
		for gen in range(1, self.iterations + 1):
			ants = [Ant(self, graph) for i in range(self.ants)]
			best_cost_local = 0
			best_local = []
			ant_local = None

			worst_cost_local = 99999

			self.build_solutions(graph, ants)
			valids = []
			for ant in ants:

				if ant.valid_path:
					valids.append(ant.total_cost)
					# worst cost local
					if ant.total_cost < worst_cost_local:
						worst_cost_local = ant.total_cost

					# worst cost global
					if ant.total_cost < worst_cost:
						worst_cost = ant.total_cost

				# best cost local
				if ant.total_cost > best_cost_local:
					best_cost_local = ant.total_cost
					best_local = [] + ant.path
					ant_local = ant

				# best cost global
				if ant.total_cost > best_cost:
					best_cost = ant.total_cost
					best_solution = [] + ant.path
					ant_global = ant

			stats.append((repetition, gen, best_cost, worst_cost, best_cost_local,
						  worst_cost_local, np.mean(valids)))

			if self.update_pheromone == 1:	# elitist
				self.pheromone_evaporation(graph)
				self.update_pheromone(graph, ants)
				self.elitist(graph, best_solution)
			elif self.update_pheromone == 2:	# max_min
				self.max_min(graph, ant_local)
			else:
				self.pheromone_evaporation(graph)
				self.update_pheromone(graph, ants)

		return best_solution, best_cost, stats


class Ant(object):

	def __init__(self, aco, graph):

		self.colony = aco
		self.graph = graph
		self.total_cost = 0.0
		self.path = [] # path list
		self.valid_path = True
		self.allowed = [i for i in range(graph.n)]
		start = 0 # start from any node
		self.path.append(start)
		self.current = start
		self.allowed.remove(start)

	def select_next(self):

		denominator = 0
		for k in self.allowed:
			if self.graph.matrix[self.current][k] != -1:
				denominator += self.graph.pheromone[self.current][k] ** self.colony.alpha * self.graph.eta[self.current][k] ** self.colony.beta

		probabilities = [0 for i in range(self.graph.n)] # probabilities for moving to a node in the next step

		for i in range(self.graph.n):
			if self.graph.matrix[self.current][i] != -1:
				try:
					self.allowed.index(i) # test if allowed list contains i
					probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
										self.graph.eta[self.current][i] ** self.colony.beta / denominator
				except ValueError:
					pass

		# select next node by probability
		selected = -1
		rand = random.random()
		for i, probability in enumerate(probabilities):
			rand -= probability
			if rand <= 0:
				selected = i
				break
		if selected != -1:
			self.allowed.remove(selected)
			self.path.append(selected)
			self.total_cost += self.graph.matrix[self.current][selected]
			self.current = selected

		return selected









