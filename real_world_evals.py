import bnlearn as bn
from testbed import *
import random

# Import DAG
np.random.seed(0)
random.seed(0)


def lower_case_nodes(graph):
	mapping = {}
	nodes = list(graph.nodes())
	for node in nodes:
		mapping[node] = str(node.lower()).replace("_", "")
	return nx.relabel_nodes(graph, mapping)


def transform_df(df):
	algos = []
	datasets = []
	times = []
	costs = []
	ratios = []
	for key in df.keys():
		algos.append(key.split('_')[0])
		datasets.append(key.split('_')[1])
		times.append(df[key][0])
		costs.append(df[key][1])
		ratios.append(df[key][2])

	cols = ['algorithm', 'dataset', 'time', 'cost', 'ratio']
	results = pd.DataFrame([algos, datasets, times, costs, ratios]).T
	results.columns = cols
	return results


def assign_weights_and_unobs(graph):
	nodes = list(graph.nodes)

	edges = graph.edges()
	for edge in edges:
		from_ = edge[0]
		to_ = edge[1]
		graph[from_][to_]['weight'] = np.random.uniform(0.51, 1)

	random.shuffle(nodes)
	num_nodes = len(nodes)
	num_unobs = np.random.randint(1, int(num_nodes * (num_nodes - 1) / 2))  # at least one unobserved var for admg
	num_unobs = int(2 * (np.log(num_nodes) / num_nodes) * num_unobs)

	pairs = [comb for comb in combinations(nodes, 2)]
	unobs_links = [pairs[i] for i in range(num_unobs)]
	for i, nodes in enumerate(unobs_links):
		from_ = nodes[0]
		to_ = nodes[1]
		u_name = 'U{}'.format(i)
		weight = np.random.uniform(0.51, 1)
		graph.add_edge(u_name, from_, weight=weight)
		graph.add_edge(u_name, to_, weight=weight)
	return graph


def get_outcome(digraph):
	checker = digraph.copy()
	checker.remove_nodes_from(list(nx.isolates(checker)))  # removes unconnected nodes (Y should have some causes)
	ordered = list(nx.topological_sort(checker))
	return {str(ordered[-1])}


def probs_to_costs(graph):
	np.seterr(divide='ignore')
	for u, v, d in graph.edges(data=True):
		w = np.float32(d['weight'])
		d['weight'] = np.log(w / (1 - w))
	return graph


def get_probs(cutset, digraph, bigraph):
	di_edges = digraph.edges(data=True)
	di_log_probs = []
	for edge in di_edges:
		di_cost = edge[2]['weight']
		di_log_probs.append(np.log(np.exp(di_cost) / (1 + np.exp(di_cost))))

	bi_edges = bigraph.edges(data=True)
	bi_log_probs = []
	for edge in bi_edges:
		bi_cost = edge[2]['weight']
		bi_log_probs.append((np.log(np.exp(bi_cost) / (1 + np.exp(bi_cost)))))
	di_log_probs.extend(bi_log_probs)

	before_log_sum = np.sum(di_log_probs)

	digraph_ = digraph.copy()
	bigraph_ = bigraph.copy()

	cutset_probs = []
	for edge in list(cutset):
		if len(edge) == 3:
			from_ = edge[0]
			to_ = edge[1]
			dibi = edge[2][0]

		else:
			dibi = edge.split('_')[0]
			from_ = edge.split('_')[1]
			to_ = edge.split('_')[2]

		if dibi == 'd':
			cost = digraph_.get_edge_data(from_, to_)['weight']
			digraph_.remove_edge(from_, to_)

		elif dibi == 'b':
			cost = bigraph_.get_edge_data(from_, to_)['weight']
			bigraph_.remove_edge(from_, to_)

		cutset_probs.append((np.exp(cost) / (1 + np.exp(cost))))

	di_edges = digraph_.edges(data=True)
	di_log_probs = []
	for edge in di_edges:
		di_cost = edge[2]['weight']
		di_log_probs.append(np.log(np.exp(di_cost) / (1 + np.exp(di_cost))))

	bi_edges = bigraph_.edges(data=True)
	bi_log_probs = []
	for edge in bi_edges:
		bi_cost = edge[2]['weight']
		bi_log_probs.append((np.log(np.exp(bi_cost) / (1 + np.exp(bi_cost)))))
	di_log_probs.extend(bi_log_probs)

	after_log_sum = np.sum(di_log_probs)

	return before_log_sum, after_log_sum, (np.log(1 - np.asarray(cutset_probs))).sum()


def run_brute(algorithm, digraph, bigraph, Y, Q, upper_bound):
	start = time.time()
	cutset, cost = edgeIDbrutev2(digraph=digraph, bigraph=bigraph, Y=Y, upper_bound=upper_bound)
	stop = time.time()
	time_taken = stop - start
	Q.put((time_taken, cost, cutset))


def run_algorithm_bespoke_graph(algorithm, digraph, bigraph, Y, Q):
	print(algorithm)
	if algorithm == 'heuristicEdgeId1':  # for single nodes
		admg_ = ADMG(digraph.copy(), bigraph.copy())
		start = time.time()
		cutset, cost = heuristicEdgeId1(admg_, Y)  # only works for single Y
		stop = time.time()
		time_taken = stop - start

	elif algorithm == 'heuristicEdgeId2':
		admg_ = ADMG(digraph.copy(), bigraph.copy())
		start = time.time()
		cutset, cost = heuristicEdgeId2(admg_, Y)  # only works for single Y
		stop = time.time()
		time_taken = stop - start

	elif algorithm == 'heuristicMinCut1':
		start = time.time()
		h_dir, h_bi, y_hat = edgeIDtoIntervention(digraph.copy(), bigraph.copy(), Y)
		h = ADMG(h_dir, h_bi)
		cutset, cost = heuristicMinCut1(h, y_hat)
		stop = time.time()
		time_taken = stop - start

	elif algorithm == 'heuristicMinCut2':
		start = time.time()
		h_dir, h_bi, y_hat = edgeIDtoIntervention(digraph.copy(), bigraph.copy(), Y)
		h = ADMG(h_dir, h_bi)
		cutset, cost = heuristicMinCut2(h, y_hat)
		stop = time.time()
		time_taken = stop - start

	elif algorithm == 'Alg2':
		start = time.time()
		h_dir, h_bi, y_hat = edgeIDtoIntervention(digraph.copy(), bigraph.copy(), Y)
		h = ADMG(h_dir, h_bi)
		cutset, cost, _ = Alg2(h, y_hat)
		stop = time.time()
		time_taken = stop - start

	Q.put((time_taken, cost, cutset))


def get_barley(barley_model):
	flag = 0
	while not flag:
		barley_model = probs_to_costs(assign_weights_and_unobs(lower_case_nodes(barley_model)))
		barley_digraph, barley_bigraph = multigraph_to_digraph_graph(barley_model)
		barley_Y = get_outcome(barley_digraph)
		flag = check_id(digraph=barley_digraph, bigraph=barley_bigraph, Y=barley_Y)
	return barley_digraph, barley_bigraph, barley_Y


def get_alarm(alarm_model):
	flag = 0
	while not flag:
		alarm_model = probs_to_costs(assign_weights_and_unobs(lower_case_nodes(alarm_model)))
		alarm_digraph, alarm_bigraph = multigraph_to_digraph_graph(alarm_model)
		alarm_Y = get_outcome(alarm_digraph)
		flag = check_id(digraph=alarm_digraph, bigraph=alarm_bigraph, Y=alarm_Y)
	return alarm_digraph, alarm_bigraph, alarm_Y


def get_water(water_model):
	flag = 0
	while not flag:
		water_model = probs_to_costs(assign_weights_and_unobs(lower_case_nodes(water_model)))
		water_digraph, water_bigraph = multigraph_to_digraph_graph(water_model)
		water_Y = get_outcome(water_digraph)
		flag = check_id(digraph=water_digraph, bigraph=water_bigraph, Y=water_Y)
	return water_digraph, water_bigraph, water_Y


def get_psygraph():
	flag = 0
	while not flag:
		psygraph = nx.DiGraph()
		psygraph.add_edge('genderR', 'supportR', weight=1.0)
		psygraph.add_edge('genderS', 'supportR', weight=1.0)
		psygraph.add_edge('advrsR', 'supportR', weight=1.0)
		psygraph.add_edge('advrsS', 'supportR', weight=1.0)
		psygraph.add_edge('distressR', 'supportR', weight=1.0)
		psygraph.add_edge('distressS', 'supportR', weight=1.0)

		psygraph.add_edge('ageR', 'supportS', weight=1.0)
		psygraph.add_edge('ageS', 'supportS', weight=1.0)
		psygraph.add_edge('genderR', 'supportS', weight=1.0)
		psygraph.add_edge('genderS', 'supportS', weight=1.0)
		psygraph.add_edge('advrsR', 'supportS', weight=1.0)
		psygraph.add_edge('advrsS', 'supportS', weight=1.0)
		psygraph.add_edge('distressR', 'supportS', weight=1.0)
		psygraph.add_edge('distressS', 'supportS', weight=1.0)

		psygraph.add_edge('ageS', 'rdciS', weight=1.0)
		psygraph.add_edge('genderR', 'rdciS', weight=1.0)
		psygraph.add_edge('genderS', 'rdciS', weight=1.0)
		psygraph.add_edge('reltype', 'rdciS', weight=1.0)
		psygraph.add_edge('advrsS', 'rdciS', weight=1.0)
		psygraph.add_edge('distressR', 'rdciS', weight=1.0)
		psygraph.add_edge('supportR', 'rdciS', weight=1.0)
		psygraph.add_edge('distressS', 'rdciS', weight=1.0)
		psygraph.add_edge('dcidydR', 'rdciS', weight=1.0)
		psygraph.add_edge('sdciR', 'rdciS', weight=1.0)
		psygraph.add_edge('dcidydS', 'rdciS', weight=1.0)
		psygraph.add_edge('cohablen', 'rdciS', weight=1.0)

		psygraph.add_edge('ageS', 'sdciS', weight=1.0)
		psygraph.add_edge('genderR', 'sdciS', weight=1.0)
		psygraph.add_edge('genderS', 'sdciS', weight=1.0)
		psygraph.add_edge('reltype', 'sdciS', weight=1.0)
		psygraph.add_edge('advrsS', 'sdciS', weight=1.0)
		psygraph.add_edge('advrsR', 'sdciS', weight=1.0)
		psygraph.add_edge('distressR', 'sdciS', weight=1.0)
		psygraph.add_edge('supportR', 'sdciS', weight=1.0)
		psygraph.add_edge('supportS', 'sdciS', weight=1.0)
		psygraph.add_edge('distressS', 'sdciS', weight=1.0)
		psygraph.add_edge('rdciR', 'sdciS', weight=1.0)
		psygraph.add_edge('dcidydR', 'sdciS', weight=1.0)
		psygraph.add_edge('dcidydS', 'sdciS', weight=1.0)
		psygraph.add_edge('cohablen', 'sdciS', weight=1.0)

		psygraph.add_edge('rdciS', 'relsatR', weight=1.0)
		psygraph.add_edge('rdciR', 'relsatR', weight=1.0)

		psygraph.add_edge('sdciS', 'relsatS', weight=1.0)
		psygraph.add_edge('dcidydS', 'relsatS', weight=1.0)

		psygraph.add_edge('ageS', 'depR', weight=1.0)
		psygraph.add_edge('genderR', 'depR', weight=1.0)
		psygraph.add_edge('genderS', 'depR', weight=1.0)
		psygraph.add_edge('advrsR', 'depR', weight=1.0)
		psygraph.add_edge('distressR', 'depR', weight=1.0)
		psygraph.add_edge('rdciS', 'depR', weight=1.0)
		psygraph.add_edge('dcidydR', 'depR', weight=1.0)
		psygraph.add_edge('sdciS', 'depR', weight=1.0)
		psygraph.add_edge('sdciR', 'depR', weight=1.0)
		psygraph.add_edge('dcidydS', 'depR', weight=1.0)

		psygraph.add_edge('ageS', 'depS', weight=1.0)
		psygraph.add_edge('genderR', 'depS', weight=1.0)
		psygraph.add_edge('genderS', 'depS', weight=1.0)
		psygraph.add_edge('advrsR', 'depS', weight=1.0)
		psygraph.add_edge('distressR', 'depS', weight=1.0)
		psygraph.add_edge('supportR', 'depS', weight=1.0)
		psygraph.add_edge('distressS', 'depS', weight=1.0)
		psygraph.add_edge('supportS', 'depS', weight=1.0)
		psygraph.add_edge('rdciS', 'depS', weight=1.0)
		psygraph.add_edge('rdciR', 'depS', weight=1.0)
		psygraph.add_edge('dcidydR', 'depS', weight=1.0)
		psygraph.add_edge('sdciS', 'depS', weight=1.0)
		psygraph.add_edge('sdciR', 'depS', weight=1.0)
		psygraph.add_edge('dcidydS', 'depS', weight=1.0)
		psygraph.add_edge('cohablen', 'depS', weight=1.0)
		psygraph.add_edge('relsatR', 'depS', weight=1.0)

		psygraph = probs_to_costs(assign_weights_and_unobs(lower_case_nodes(psygraph)))
		digraph_psy, bigraph_psy = multigraph_to_digraph_graph(psygraph)
		psy_Y = get_outcome(digraph_psy)
		flag = check_id(digraph=digraph_psy, bigraph=bigraph_psy, Y=psy_Y)
	return digraph_psy, bigraph_psy, psy_Y


if __name__ == '__main__':

	fnbar = 'bnlearn_graphs/barley.bif'
	fnalarm = 'bnlearn_graphs/alarm.bif'
	fnwater = 'bnlearn_graphs/water.bif'
	barley_model = nx.from_pandas_adjacency(bn.import_DAG(fnbar)['adjmat'], create_using=nx.DiGraph())
	alarm_model =  nx.from_pandas_adjacency(bn.import_DAG(fnalarm)['adjmat'], create_using=nx.DiGraph())
	water_model = nx.from_pandas_adjacency(bn.import_DAG(fnwater)['adjmat'], create_using=nx.DiGraph())

	graphtypes = ['psych', 'barley', 'water', 'alarm']
	algorithms = ['heuristicEdgeId1', 'heuristicEdgeId2', 'heuristicMinCut1',
	              'heuristicMinCut2', 'Alg2']

	results_dict = {}

	for i in range(len(graphtypes)):
		graphtype = graphtypes[i]
		print(graphtype)
		if graphtype == 'psych':
			digraph, bigraph, Y = get_psygraph()

		elif graphtype == 'barley':
			digraph, bigraph, Y = get_barley(barley_model)

		elif graphtype == 'water':
			digraph, bigraph, Y = get_water(water_model)

		elif graphtype == 'alarm':
			digraph, bigraph, Y = get_alarm(alarm_model)

		for algorithm in algorithms:
			res_name = algorithm + '_' + graphtype
			Q = multiprocessing.Queue()
			# RUN ALGORITHM
			p = multiprocessing.Process(target=run_algorithm_bespoke_graph, args=(algorithm, digraph, bigraph, Y, Q))
			p.start()

			start = time.time()

			kill = False
			finished = False
			check_interval = 0.001
			timeout_lim = 500 if 'heuristic' not in algorithm else 10000

			while not kill and not finished:
				time.sleep(check_interval)
				now = time.time()
				runtime = now - start
				if not p.is_alive():
					time_taken, cost, cutset = Q.get()
					if cost != 0:
						before_logsum, after_logsum, cutset_invlogsum = get_probs(cutset, digraph, bigraph)
						ratio = np.exp((after_logsum + cutset_invlogsum)) / np.exp(before_logsum)
					else:
						ratio = 1.0
					results_dict[res_name] = (time_taken, cost, ratio)
					finished = True
					p.join()

				elif runtime > timeout_lim:
					print('Took too long.')
					kill = True
					p.terminate()
					p.join()
					results_dict[res_name] = (np.inf, np.inf, np.inf)

	results_dict = transform_df(results_dict)

	# reset random seed to make sure we evaluate on the same graphs!
	np.random.seed(0)
	random.seed(0)

	graphtypes = ['psych', 'barley', 'water', 'alarm']

	results_dict_brute = {}
	for i in range(len(graphtypes)):
		graphtype = graphtypes[i]
		print(graphtype)
		if graphtype == 'psych':
			digraph, bigraph, Y = get_psygraph()

		elif graphtype == 'barley':
			digraph, bigraph, Y = get_barley(barley_model)

		elif graphtype == 'water':
			digraph, bigraph, Y = get_water(water_model)

		elif graphtype == 'alarm':
			digraph, bigraph, Y = get_alarm(alarm_model)

		HEID1_upper_bound = results_dict[
			(results_dict.algorithm == 'heuristicEdgeId1') & (results_dict.dataset == graphtype)].cost.values
		HEID2_upper_bound = results_dict[
			(results_dict.algorithm == 'heuristicEdgeId2') & (results_dict.dataset == graphtype)].cost.values
		heurist_upper_bound = min(np.array([HEID1_upper_bound[0], HEID2_upper_bound[0]]))

		res_name = algorithm + '_' + graphtype
		Q = multiprocessing.Queue()
		# RUN ALGORITHM
		p = multiprocessing.Process(target=run_brute, args=(algorithm, digraph, bigraph, Y, Q, heurist_upper_bound))
		p.start()

		start = time.time()

		kill = False
		finished = False
		check_interval = 0.001
		timeout_lim = 500

		while not kill and not finished:
			time.sleep(check_interval)
			now = time.time()
			runtime = now - start
			if not p.is_alive():
				time_taken, cost, cutset = Q.get()
				if cost != 0:
					before_logsum, after_logsum, cutset_invlogsum = get_probs(cutset, digraph, bigraph)
					ratio = np.exp((after_logsum + cutset_invlogsum)) / np.exp(before_logsum)
				else:
					ratio = 1.0
				results_dict_brute[res_name] = (time_taken, cost, ratio)
				finished = True
				p.join()

			elif runtime > timeout_lim:
				print('Took too long.')
				kill = True
				p.terminate()
				p.join()
				results_dict_brute[res_name] = (np.inf, np.inf, np.inf)

	print()




