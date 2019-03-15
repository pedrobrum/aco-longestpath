#!/usr/bin/env python3

import math
import sys
import collections
import time
import argparse
import pandas as pd
from aco import ACO, Graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", required=True,
                        help="graph file.")
    parser.add_argument("-a", "--num_ants", type=int, default=200,
                        help="number of ants.")
    parser.add_argument("-i", "--iterations", type=int, default=100,
                        help="number of iterations.")
    parser.add_argument("-e", "--evaporation", type=float, default=0.2,
                        help="evaporation rate of aco.")
    parser.add_argument("-pi", "--intensity", type=float, default=0.5,
                        help="pheromone intensity.")
    parser.add_argument("-ip", "--initial", type=float, default=1.0,
                        help="initial pheromone.")
    parser.add_argument("--alpha", type=int, default=1.0,
                        help="alpha value.")
    parser.add_argument("--beta", type=int, default=4.0,
                        help="beta value.") 
    parser.add_argument("--runs", type=int, default=10,
                        help="number of executions.")
    parser.add_argument("-o", "--output_file", required=True,
                        help="output csv to be written.")

    return parser.parse_args()


def read_file(file):

    edges = []
    graph = collections.defaultdict(list)
    with open(file) as f:
        for line in f.readlines():
            line = line.split('\t')
            x = int(line[0])
            y = int(line[1])
            w = int(line[2])
            edges.append((x, y, w))
            graph[x].append((y, w))

    n = len(graph)
    m = len(edges)

    print('Number of nodes: ', n)
    print('Number of edges: ', m)


    # initializes matrix
    matrix = [[-1]*n for i in range(n)]

    # add edges
    for i in range(0, len(edges)):
        t = edges[i]
        x = t[0] - 1
        y = t[1] - 1
        w = t[2]
        matrix[x][y] = w

    return matrix, n, m


def main():

    args = parse_args()

    file = args.graph
    matrix, n, m = read_file(file)

    # execution parameters
    ants = args.num_ants
    iterations = args.iterations
    evaporation = args.evaporation
    alpha = args.alpha
    beta = args.beta
    runs = args.runs

    # default
    strategy = 1
    update_pheromone = 2
    min_ph = 0.0001
    max_ph = 10.0
    initial_pheromone = args.initial
    q = args.intensity

    final_results = []
    train_results = []
    for i in range(1, runs + 1):
        # create ACO object
        aco = ACO(ants, iterations, alpha, beta, evaporation, q, strategy,
                  update_pheromone, min_ph, max_ph)
        # create Graph object
        graph = Graph(matrix, n, m, initial_pheromone)
        # Run ACO
        path, cost, stats = aco.solve(graph, i)
        print('------------------------------')
        print('Run ', i)
        print('cost: {}, path: {}'.format(cost, path))
        print(len(path))
        final_results.append((i, cost, len(path)))
        train_results = train_results + stats


    output_file = args.output_file

    col_names = ("repetition", "iteration", "best_cost", "worst_cost", "best_local",
                 "worst_local", "mean_local")

    frame = pd.DataFrame.from_records(train_results, columns=col_names)
    frame.to_csv(output_file + "_train.csv", index=False, sep='\t', encoding='utf-8')

    col_names = ("repetition", "cost", "path_len")
    frame = pd.DataFrame.from_records(final_results, columns=col_names)
    frame.to_csv(output_file + "_final.csv", index=False, sep='\t', encoding='utf-8')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % ((time.time() - start_time)))

