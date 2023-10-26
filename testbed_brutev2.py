from ADMG_Sampler import *
from edgeIDv3 import *
import networkx as nx
import time
import pandas as pd
import numpy as np
import os
import multiprocessing
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

        # rename algorithms
        if alg == 'heuristicEdgeId1':
            alg = 'HEID-1'
        elif alg == 'heuristicEdgeId2':
            alg = 'HEID-2'
        elif alg == 'heuristicMinCut1':
            alg = 'MCIP-H1'
        elif alg == 'heuristicMinCut2':
            alg = 'MCIP-H2'
        elif alg == 'Alg2':
            alg = 'MCIP-exact'
        elif alg == 'edgeIDbrutev3':
            alg = 'EDGEID'

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

def sample_graph_outcome(library, set_inf):
    # Sample graph from librarydigraph, bigraph = get_graphs(graph)

    while True:  # Assign edge probabilities and find combination which is identifiable in principle
        dlen = 0
        ulen = 0

        while (dlen < 1 or ulen < 1):  # prevents sampling empty digraph
            graph = random.choice(library)
            digraph, bigraph = multigraph_to_digraph_graph(graph)
            dlen = len(digraph.edges)
            ulen = len(bigraph.edges)

        checker = digraph.copy()
        checker.remove_nodes_from(list(nx.isolates(checker)))  # removes unconnected nodes (Y should have some causes)
        ordered = list(nx.topological_sort(checker))
        Y = {str(ordered[-1])}
        proba_graph = ds.edge_weighting(graph=graph.copy(), costs=cost_setting, rounding=rounding)
        digraph, bigraph = multigraph_to_digraph_graph(proba_graph)

        if set_inf:
            digraph = set_to_inf(graph=digraph.copy(), q=0.5)
            bigraph = set_to_inf(graph=bigraph.copy(), q=0.5)
        nodes = list(digraph.nodes)
        for node in nodes:
            nx.relabel_nodes(digraph, {node: str(node)}, copy=False)

        nodes = list(bigraph.nodes)
        for node in nodes:
            nx.relabel_nodes(bigraph, {node: str(node)}, copy=False)

        if check_id(digraph=digraph, bigraph=bigraph, Y=Y):
            print('Viable graph, running algorithm(s)...')
            break
        else:
            print('No solution, sampling another set of edge weights.')

    return graph, proba_graph, digraph, bigraph, Y

def save_graph(digraph, bigraph, fn):
    d_list = list(digraph.edges(data=True))
    d_list = [tuple(list(g) + ['di']) for g in d_list]

    b_list = list(bigraph.edges(data=True))
    b_list = [tuple(list(g) + ['bi']) for g in b_list]
    d_list.extend(b_list)

    froms = []
    tos = []
    bidis = []
    weights = []
    for edge in d_list:
        froms.append(edge[0])
        tos.append(edge[1])
        bidis.append(edge[-1])
        weights.append(edge[-2]['weight'])

    edge_df = pd.DataFrame([bidis, froms, tos, weights]).T
    edge_df.columns = ['di or bi', 'from_node', 'to_node', 'weight']
    edge_df.to_csv('graphs/graph_{}.csv'.format(fn), index=False)


def plot_graph(graph, directed, weights=False):
    # function for plotting a graph (directed or undirected)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size=500, with_labels=True, arrows=directed, connectionstyle='arc3, rad = 0.1')
    if weights:
        labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in graph.edges(data=True)])
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


def get_finite_edges(graph):
    edges = list(graph.edges(data=True))
    to_remove = []
    for edge in edges:
        from_ = edge[0]
        to_ = edge[1]
        weight = edge[2]['weight']
        if weight != np.inf:
            to_remove.append((from_, to_))
    return to_remove


def check_id(digraph, bigraph, Y):
    MH = hedgeHull2(digraph=digraph, bigraph=bigraph, S=Y)
    hedge_digraph = (nx.subgraph(digraph, MH)).copy()
    hedge_bigraph = (nx.subgraph(bigraph, MH)).copy()

    digraph_remove = get_finite_edges(hedge_digraph)
    bigraph_remove = get_finite_edges(hedge_bigraph)

    for edge in digraph_remove:
        hedge_digraph.remove_edge(*edge)
    for edge in bigraph_remove:
        hedge_bigraph.remove_edge(*edge)

    return isIdentifiable2(digraph=hedge_digraph, bigraph=hedge_bigraph, S=Y, H=MH)


def sample_graph_outcome(library, set_inf):
    # Sample graph from librarydigraph, bigraph = get_graphs(graph)

    while True:  # Assign edge probabilities and find combination which is identifiable in principle
        dlen = 0
        ulen = 0

        while (dlen < 1 or ulen < 1):  # prevents sampling empty digraph
            graph = random.choice(library)
            digraph, bigraph = multigraph_to_digraph_graph(graph)
            dlen = len(digraph.edges)
            ulen = len(bigraph.edges)

        checker = digraph.copy()
        checker.remove_nodes_from(list(nx.isolates(checker)))  # removes unconnected nodes (Y should have some causes)
        ordered = list(nx.topological_sort(checker))
        Y = {str(ordered[-1])}
        proba_graph = ds.edge_weighting(graph=graph.copy(), costs=cost_setting, rounding=rounding)
        digraph, bigraph = multigraph_to_digraph_graph(proba_graph)

        if set_inf:
            digraph = set_to_inf(graph=digraph.copy(), q=0.5)
            bigraph = set_to_inf(graph=bigraph.copy(), q=0.5)
        nodes = list(digraph.nodes)
        for node in nodes:
            nx.relabel_nodes(digraph, {node: str(node)}, copy=False)

        nodes = list(bigraph.nodes)
        for node in nodes:
            nx.relabel_nodes(bigraph, {node: str(node)}, copy=False)

        if check_id(digraph=digraph, bigraph=bigraph, Y=Y):
            print('Viable graph, running algorithm(s)...')
            break
        else:
            print('No solution, sampling another set of edge weights.')

    return graph, proba_graph, digraph, bigraph, Y


def set_to_inf(graph, q):
    for u, v, d in graph.edges(data=True):
        make_inf = random.uniform(0, 1)
        if make_inf > q:
            d['weight'] = np.inf
    return graph


def run_brute(digraph, bigraph, upper_bound):
    print('Running Brute v3')
    start = time.time()
    _, cost = edgeIDbrutev3(digraph=digraph, bigraph=bigraph, Y=Y, upper_bound=upper_bound)
    stop = time.time()
    time_taken = stop - start

    Q.put((time_taken, cost))


if __name__ == '__main__':
    # load previous results
    fn = 'all_results/v3_all_noinfpen_nosparsity_rerun.csv'  # results of heuristic algorithms to use as upper bounds

    res_fn = 'edgebrute3UB.csv'
    algorithms = ['edgeIDbrutev3UB']

    df = pd.read_csv(fn)

    df.columns = ['run_details', 'runtime cost']
    df = reformat_df(df)

    # Graph search params:
    admg = True  # sample probabilistic graph
    epsilon = 0.1  # new graph discovery rate threshold
    seed = 0
    max_graphs = 20  # maximum desired number of canonical ADMGs for a given number of nodes
    max_iters = 100  # number of iters when searching for graphs

    # Experiment params:
    rounding = False  # whether to quantise the edge probability weights to nexarest decimal  (boolean)
    cost_setting = True  # whether to use the weights in log(pe / (1-pe)) form
    sims = 50  # number of graphs to evaluate
    sims_per_graph = 1  # number of averages over randomly sampled edge weights
    # num_nodes_ = [20, 30, 40, 50, 80, 100, 150, 200, 250]  # (observed)
    num_nodes_ = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250]
    num_outcomes = 1
    set_inf = False  # whether to set edge weights to inf with a probability 0.5
    sparsity = False  # whether to enforce a sparsity contrain to reduce the density of the sampled graphs


    # algorithms = ['NaiveGreedy']

    verbose = False

    graph_folder = os.path.join('all_results', 'graphs')
    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)

    results_dict = {}

    for num_nodes in num_nodes_:
        print('Num Nodes:', num_nodes)
        noniso = True if num_nodes < 200 else False  # whether to only produce graphs which are non-isomorphisms of each other (boolean)

        if sparsity:
            sparsity_param = np.log(
                num_nodes) / num_nodes  # encourages sparsity in the canonical library as the number of nodes increases
        else:
            sparsity_param = None
        # Initialise DAGSampler object
        ds = DAGSampler(library=None, num_nodes=num_nodes, admg=admg, seed=seed)
        # Get canonical library
        library = ds.generate_library(plot=False, verbose=verbose, max_iters=max_iters, sparsity_param=sparsity_param,
                                      epsilon=epsilon, max_graphs=max_graphs, nonisomorphic=noniso)

        g_sims = []
        t = 0
        for g in range(sims):  # start sampling graphs with num_nodes

            graph, proba_graph, digraph, bigraph, Y = sample_graph_outcome(library=library, set_inf=set_inf)

            p_sims = []
            costs = []

            save_graph(digraph=digraph, bigraph=bigraph, fn=str(num_nodes) + '_' + str(t))

            # pull previous minimum cost found from HEID algorithms
            prev_min_result = min(df[(df.num_nodes == num_nodes) & (df.simulation == g) & (
                        (df.algorithm == 'HEID-1') | (df.algorithm == 'HEID-2'))]['cost'].values)
            print('Using {} upper bound.'.format(prev_min_result))

            for algorithm in algorithms:

                # print('Testing algorithm:', algorithm)
                eval_name = 'algorithm_' + algorithm + '_num_nodes_' + str(num_nodes).zfill(6) + '_sim_' + str(
                    g).zfill(3)

                Q = multiprocessing.Queue()
                # RUN ALGORITHM
                p = multiprocessing.Process(target=run_brute, args=(digraph, bigraph, prev_min_result))
                p.start()

                start = time.time()

                kill = False
                finished = False
                check_interval = 0.001
                timeout_lim = 180 if 'heuristic' not in algorithm else 10000

                while not kill and not finished:
                    time.sleep(check_interval)
                    now = time.time()
                    runtime = now - start
                    if not p.is_alive():
                        time_taken, cost = Q.get()
                        results_dict[eval_name] = (time_taken, cost)
                        finished = True
                        p.join()

                    elif runtime > timeout_lim:
                        print('Took too long.')
                        kill = True
                        p.terminate()
                        p.join()
                        results_dict[eval_name] = (np.inf, np.inf)
            t += 1

    results = pd.DataFrame(results_dict.items())
    results.columns = ['run_details', 'runtime']
    results.to_csv('all_results/' + res_fn, index=False)





