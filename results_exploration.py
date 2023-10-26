from ADMG_Sampler import *
from edgeIDv3 import *
import pandas as pd
import numpy as np
import re


def reformat_df(df):
	algs = []
	num_nodes = []
	sims = []
	edge_vars = []
	times = []
	costs = []
	for run in df.run_details.values:
		alg = run.split('_')[1]
		num_node = int(run.split('_')[4])
		sim = int(run.split('_')[6])
		edge_var = int(run.split('_')[-1])
		time = float(re.split(',|\(|\)', df[df.run_details == run]['runtime cost'].values[0])[1])
		cost = float(re.split(',|\(|\)', df[df.run_details == run]['runtime cost'].values[0])[2])

		algs.append(alg)
		num_nodes.append(num_node)
		sims.append(sim)
		edge_vars.append(edge_var)
		times.append(time)
		costs.append(cost)

	cols = ['algorithm', 'num_nodes', 'simulation', 'edge_weight_variant', 'time', 'cost']
	df_reformat = pd.DataFrame([algs, num_nodes, sims, edge_vars, times, costs]).T
	df_reformat.columns = cols
	return df_reformat


def get_av_se_times_costs(df):
	av_times = []
	se_times = []
	av_costs = []
	se_costs = []
	for num_node in num_nodes:
		times = np.ma.masked_invalid(np.asarray(df[df.num_nodes == num_node]['time'].values, dtype=float))
		costs = np.ma.masked_invalid(np.asarray(df[df.num_nodes == num_node]['cost'].values, dtype=float))
		av_times.append(times.mean())
		se_times.append(times.std())
		av_costs.append(np.ma.masked_invalid(costs).mean())
		se_costs.append(costs.std())
	return av_times, se_times, av_costs, se_costs

if __name__ == '__main__':

	algorithms = ['heuristicEdgeId1', 'heuristicEdgeId2', 'heuristicMinCut1', 'heuristicMinCut2', 'Alg2', 'NaiveGreedy']

	fn1 = 'all_results/heuristics.csv'  # results of heuristic algorithms
	fn2 = 'all_results/exact.csv'  # results of more expensive algorithms
	fn3 = 'all_results/naivegreedy.csv'  # results of more expensive algorithms

	df1 = pd.read_csv(fn1)
	df2 = pd.read_csv(fn2)
	df3 = pd.read_csv(fn3)

	df1.columns = ['run_details', 'runtime cost']
	df1 = reformat_df(df1)
	df1 = df1[df1.num_nodes < 300]

	df2.columns = ['run_details', 'runtime cost']
	df2 = reformat_df(df2)

	df3.columns = ['run_details', 'runtime cost']
	df3 = reformat_df(df3)

	# concatenate the two together
	df = pd.concat([df2, df1, df3])

	df = df[df.algorithm != 'edIDbrute']

	markers = ['s', 'o', 'v', '+', '^', 'p', 'X']
	colors = ['r', 'g', 'b', 'c', 'k', 'm', 'darkorchid']
	cp = 5
	plt.figure(figsize=(10, 6))
	plt.rcParams.update({'font.size': 13})
	for i, algorithm in enumerate(algorithms):
		marker = markers[i]
		c = colors[i]
		data = df[df.algorithm == algorithm]
		num_nodes = np.unique(data.num_nodes)
		av_times, se_times, av_costs, se_costs = get_av_se_times_costs(data)

		se_times = 0 if algorithm == 'Alg2' else se_times
		av_times = av_times[:-1] if algorithm == 'Alg2' else av_times
		num_nodes = num_nodes[:-1] if algorithm == 'Alg2' else num_nodes

		plt.errorbar(x=num_nodes, y=av_times, yerr=se_times, color=c, fmt=marker, capsize=cp, linestyle='-',
		             label=algorithm)

	plt.xlabel('Num Nodes before Y')
	plt.ylabel('Time (seconds)')
	plt.title('Runtimes')
	plt.legend()
	plt.ylim(-0.1, 10)
	plt.grid()
	plt.savefig('all_results/times.png', dpi=150)
	plt.show()

	markers = ['s', 'o', 'v', '+', '^', 'p', 'X']
	colors = ['r', 'g', 'b', 'c', 'k', 'm', 'darkorchid']
	cp = 5
	plt.figure(figsize=(10, 6))
	plt.rcParams.update({'font.size': 13})
	for i, algorithm in enumerate(algorithms):
		marker = markers[i]
		c = colors[i]
		data = df[df.algorithm == algorithm]
		num_nodes = np.unique(data.num_nodes)
		av_times, se_times, av_costs, se_costs = get_av_se_times_costs(data)

		se_costs = 0 if algorithm == 'Alg2' else se_costs
		av_costs = av_costs[:-1] if algorithm == 'Alg2' else av_costs

		num_nodes = num_nodes[:-1] if algorithm == 'Alg2' else num_nodes

		plt.errorbar(x=num_nodes, y=av_costs, yerr=se_costs, color=c, fmt=marker, capsize=cp, linestyle='-',
		             label=algorithm)

	plt.xlabel('Num Nodes before Y')
	plt.ylabel('Cost')
	plt.legend()
	plt.title('Costs')
	plt.ylim(-0.1, 500)
	plt.grid()
	plt.savefig('all_results/costs.png', dpi=150)
	plt.show()

	markers = ['s', 'o', 'v', '+', '^', 'p', 'X']
	colors = ['r', 'g', 'b', 'c', 'k', 'm', 'darkorchid']
	cp = 5
	plt.figure(figsize=(10, 6))
	plt.rcParams.update({'font.size': 13})
	for i, algorithm in enumerate(algorithms):
		marker = markers[i]
		c = colors[i]
		data = df[df.algorithm == algorithm]
		num_nodes = np.unique(data.num_nodes)
		av_times, se_times, av_costs, se_costs = get_av_se_times_costs(data)

		se_costs = 0 if algorithm == 'Alg2' else se_costs
		av_costs = av_costs[:-1] if algorithm == 'Alg2' else av_costs

		num_nodes = num_nodes[:-1] if algorithm == 'Alg2' else num_nodes

		plt.errorbar(x=num_nodes, y=av_costs, yerr=se_costs, color=c, fmt=marker, capsize=cp, linestyle='-',
		             label=algorithm)

	plt.xlabel('Num Nodes before Y')
	plt.ylabel('Cost')
	plt.legend()
	plt.title('Costs (low n)')
	plt.ylim(-0.1, 30)
	plt.xlim(3, 25)
	plt.grid()
	plt.savefig('all_results/costs_zoom.png', dpi=150)
	plt.show()

	plt.figure(figsize=(10, 6))
	plt.rcParams.update({'font.size': 13})
	for i, algorithm in enumerate(algorithms):

		num_nodes_ = np.unique(df[df.algorithm == algorithm].num_nodes)

		percent_infs = []
		for num_node in num_nodes_:
			data = df[(df.num_nodes == num_node) & (df.algorithm == algorithm)]
			num_graphs = len(data)
			num_infs = ((data.cost == np.inf).sum())  # either because of time-out or because no solution found
			percent_infs.append(num_infs / num_graphs)

		marker = markers[i]
		c = colors[i]
		plt.errorbar(x=num_nodes_, y=percent_infs, color=c, capsize=cp, fmt=marker, linestyle='-',
		             label=algorithm)

	plt.xlabel('Num Nodes before Y')
	plt.ylabel('Fraction solution not found / reached time-out (> 3mins)')
	plt.ylim(-0.01, 1)
	plt.title('Failures/Timeouts')
	plt.legend()
	plt.grid()
	plt.savefig('all_results/percent_failed.png', dpi=150)
	plt.show()