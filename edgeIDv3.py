import itertools
import networkx as nx
import warnings
from collections import deque
from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from ADMG_Sampler import multigraph_to_digraph_graph
warningsOn = True

"""" Extending the Networkx functions to support multiple nodes as the bfs source:
 Note: The following functions are modified based on Networkx implementations for the purposes of this work, to allow
 for finding the set of ancestors of a set of nodes rather than only a single node.
"""


def generic_bfs_edges_general(G, source, neighbors=None, depth_limit=None, sort_neighbors=None):
    if callable(sort_neighbors):
        _neighbors = neighbors
        neighbors = lambda node: iter(sort_neighbors(_neighbors(node)))

    visited = {node for node in source}
    if depth_limit is None:
        depth_limit = len(G)
    queue = deque([(node, depth_limit, neighbors(node)) for node in source])
    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now > 1:
                    queue.append((child, depth_now - 1, neighbors(child)))
        except StopIteration:
            queue.popleft()


def bfs_edges_general(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    yield from generic_bfs_edges_general(G, source, successors, depth_limit, sort_neighbors)


# source in the input of ancestors_general is a set of nodes (a list, or a set)
# IMPORTANT NOTE: ancestors_general() includes the source itself unlike nx.ancestors()
def ancestors_general(G, source):
    return {child for parent, child in bfs_edges_general(G, source, reverse=True)}.union(source)



class ADMG:
    # initialize the admg instance with g_dir (directed graph) and g_bi (bidirected graph) over the same set of nodes.
    def __init__(self, g_dir, g_bi):
        if warningsOn:
            if g_dir.nodes != g_bi.nodes:
                warnings.warn('Mismatched node names/ Exiting.')
                return
        self.n = len(g_dir.nodes)  # number of nodes
        self.g_dir = g_dir
        self.g_bi = g_bi
        self.nodes = set(g_dir.nodes)
        self.nodeCosts = dict(g_dir.nodes(data='weight', default=np.inf))
        return

    # return the set of nodes:
    def get_nodes(self):
        return self.nodes

    # return the nodeCosts:
    def get_nodeCosts(self):
        return self.nodeCosts

    # Is a set of nodes ancestral for S?
    def isAncestral(self, S, H):
        # consider the subgraph over H:
        sub_g = nx.subgraph(self.g_dir, H)
        return H == ancestors_general(sub_g, S)

    # return the parents of S which are not in S itself:
    def parents(self, S, H=None):
        if H is None:
            dag = self.g_dir
        else:
            dag = nx.subgraph(self.g_dir, H)
        parS = []
        for s in S:
            parS += dag.predecessors(s)
        return set(parS).difference(set(S))

    # return those nodes that have a bidirected edge to at least one node in S:
    def bidir(self, S, H=None):
        if H is None:
            g = self.g_bi
        else:
            g = nx.subgraph(self.g_bi, H)
        bidirS = []
        for s in S:
            bidirS += g.neighbors(s)
        return set(bidirS).difference(set(S))

    # return the intersection of bidir and parents of S:
    def directParents(self, S, H=None):  # directParents must be intervened upon.
        return self.parents(S, H).intersection(self.bidir(S, H))

    # does a subgraph H form a hedge for S?
    def isHedge(self, S, H):
        if warningsOn:
            if not set(S).issubset(H):
                warnings.warn("Call to isHedge: S is not a subset of H!")
            if not nx.is_connected(nx.subgraph(self.latent, S)):
                warnings.warn("Call to isHedge: S is not a c-component!")

        # H forms a hedge for S iff it is ancestral and it is a c-component:
        if self.isAncestral(S, H):
            if nx.is_connected(nx.subgraph(self.g_bi, H)):
                return True
        return False

    # Construct the hedge hull of S in the subgraph H:
    def hedgeHull(self, S, H=None):
        if H is None:
            H = self.nodes  # The whole graph
        if warningsOn:
            if not S.issubset(H):
                warnings.warn("Call to hedgeHull: S is not a subset of H!")
                return None
            if not nx.is_connected(nx.subgraph(self.g_bi, S)):
                warnings.warn("Call to hedgeHull: S is not a c-component!")
                return None
        subset = copy(H)
        # Ancestor of S in H:
        anc_set = ancestors_general(nx.subgraph(self.g_dir, subset), S)
        s = list(S)[0]
        # connected component of S in anc_set:
        con_comp = nx.node_connected_component(nx.subgraph(self.g_bi, anc_set), s)
        if con_comp == subset:
            return subset
        subset = con_comp
        # Find the largest set of nodes which is ancestral for S and is a c-component:
        while True:
            anc_set = ancestors_general(nx.subgraph(self.g_dir, subset), S)
            if anc_set == subset:
                return subset
            subset = anc_set
            con_comp = nx.node_connected_component(nx.subgraph(self.g_bi, subset), s)
            if con_comp == subset:
                return subset
            subset = con_comp

    # Determine if S is identifiable in subgraph H
    def isIdentifiable(self, S, H=None):
        if H is None:
            H = self.nodes  # The whole graph
        if warningsOn:
            if not set(S).issubset(H):
                # S is not a subset of H, so not ID
                return False
        if set(S) == self.hedgeHull(S, H):
            return True
        return False

    # calculate the cost of intervention on a set I:
    def interventionCost(self, I={}):
        return sum([self.nodeCosts[i] for i in I])

    # check whether Q[S] becomes identifiable after intervention on I. return the cost of this intervention as well.
    def interventionResult(self, S, I={}):
        H = set(self.nodes).difference(I)  # intervention on I is equivalent to looking at Q[H]
        return [self.isIdentifiable(S, H), self.interventionCost(I)]

    # brute force algorithm to determine the optimal intervention to identify Q[S]:
    def optimalIntervention(self, S, H=None):
        if H is None:
            H = copy(self.nodes)
        comps = nx.connected_components(self.nodeSubgraph(S, directed=False))  # C-components of S
        comps = [c for c in comps]
        dirParents = []
        for comp in comps:
            dirParents += list(self.directParents(comp, H))
        dirParents = set(dirParents)
        # dirParents must be intervened upon.
        baseCost = sum([self.nodeCosts[i] for i in dirParents])
        if all([self.interventionResult(c, dirParents)[0] for c in comps]):  # if the set of direct Parents
            # is enough to identify
            return [dirParents, baseCost]
        HHulls = {tuple(c): self.hedgeHull(c, H.difference(dirParents)) for c in comps}  # hedge hulls of each component
        H_uni = set.union(*HHulls.values())
        minCostAdd = np.inf
        optInterv = H_uni.difference(S)
        optCosts = sorted(copy([self.nodeCosts[v] for v in optInterv]))
        for i in range(0, len(optInterv) + 1):  # all subsets
            for subset in itertools.combinations(optInterv, i):
                if sum([self.nodeCosts[i] for i in subset]) < minCostAdd:
                    I = set(subset).union(dirParents)
                    if all([self.interventionResult(c, I)[0] for c in comps]):
                        minCostAdd = sum([self.nodeCosts[i] for i in subset])
                        intervSet = I
            if minCostAdd < sum(optCosts[:i + 1]):
                break
        return [intervSet, baseCost + minCostAdd]

    # return the node with the smallest cost to intervene upon.
    def smallestCostVertex(self, H):  # return the min cost vertex among H
        costsH = {v: self.nodeCosts[v] for v in H}
        return min(costsH, key=costsH.get)

    # permanently intervene on a set I of nodes.
    def permIntervene(self, I={}):
        self.nodes = self.nodes.difference(I)
        self.g_bi = nx.subgraph(self.g_bi, self.nodes)
        self.g_dir = nx.subgraph(self.g_dir, self.nodes)
        self.nodeCosts = {v: self.nodeCosts[v] for v in self.nodes}
        return

    # return a particular subgraph over directed or bidirected edges only on the set of nodes H:
    def nodeSubgraph(self, H, directed=True):
        if directed:
            return nx.subgraph(self.g_dir, H)
        else:
            return nx.subgraph(self.g_bi, H)

    # count the number of hedges formed for Q[S] in H:
    def countHedges(self, S, H=None):  # Count the number of hedges formed for S in H
        if H is None:
            H = copy(self.nodes)
        count = 0
        if S.issubset(H):
            H = self.hedgeHull(S, H)
            if H == S:
                return 0
            HminusS = list(H.difference(S))
            count += self.countHedges(S, H.difference([HminusS[0]]))
            for i in range(1, len(HminusS)):  # all subsets
                for subset in itertools.combinations(list(set(HminusS).difference([HminusS[0]])), i):
                    I = list(set(subset).union(S).union([HminusS[0]]))
                    if self.isHedge(S, I):
                        count += 1
        else:
            warnings.warn('Call to countHedges: S is not a subset of H!')
        return count

    # plot the admg over the nodes H for the causal query Q[S]
    def plotWithNodeWeights(self, S={}, H=None):
        if H is None:
            H = set(self.g_dir.nodes)   # The whole graph
        G1 = self.nodeSubgraph(H, directed=True)
        G2_edges = self.nodeSubgraph(H, directed=False).edges
        G2 = nx.DiGraph()
        G2.add_edges_from(G2_edges)
        pos = nx.kamada_kawai_layout(G1)
        nx.draw_networkx_nodes(G1, pos, node_size=400,
                               nodelist=list(H.difference(S)),
                               node_color=None)
        nx.draw_networkx_nodes(G1, pos, node_size=400, nodelist=list(S), node_color='red')
        nx.draw_networkx_edges(G1, pos)
        nx.draw_networkx_edges(G2, pos, style=':', connectionstyle="arc3, rad=-0.4", arrowsize=0.01, edge_color='blue')
        nx.draw_networkx_labels(G1, pos)
        for i in H:
            x, y = pos[i]
            plt.text(x+0.03, y+0.03, s=str(self.nodeCosts[i]), bbox=dict(facecolor='yellow', alpha=0.5),
                     horizontalalignment='center', fontsize=7)
        plt.show()
        return


# A function that transforms an intervention instance to an edge ID instance. The input are the Directed and Bidirected
# graphs with a set of costs on nodes. The output are the corresponding Directed and Bidirected graphs with costs on
# edges. Q[Y] is the desired query to identify.
def interventionToedgeID(g_dir, g_bi, Y):
    h_dir = nx.relabel_nodes(g_dir, {i: str(i) for i in list(g_dir.nodes)})
    # duplicating each node in V\Y
    h_dir.add_nodes_from([str(i) + "'" for i in set(g_dir.nodes).difference(Y)])

    h_bi = nx.relabel_nodes(g_bi, {i: str(i) + "'" for i in set(g_bi.nodes).difference(Y)} | {i: str(i) for i in Y})
    h_bi.add_nodes_from([str(i) + "'" for i in set(g_bi.nodes).difference(Y)])

    # Edges between x' and x, which represent the vertices in g_dir and g_bi:
    h_bi.add_edges_from(zip([str(i) + "'" for i in set(g_bi.nodes).difference(Y)],
                            [str(i) for i in set(g_bi.nodes).difference(Y)]))
    # reset all the edge and node weights:
    nodes_dir = [(v, {'weight': np.inf}) for v in h_dir.nodes]
    edges_dir = [(v, w, {'weight': np.inf}) for (v, w) in h_dir.edges]
    h_dir = nx.DiGraph()
    h_dir.add_nodes_from(nodes_dir)
    h_dir.add_edges_from(edges_dir)
    nodes_bi = [(v, {'weight': np.inf}) for v in h_bi.nodes]
    edges_bi = [(v, w, {'weight': np.inf}) for (v, w) in h_bi.edges]
    h_bi = nx.Graph()
    h_bi.add_nodes_from(nodes_bi)
    h_bi.add_edges_from(edges_bi)
    # add the weight-sensitive edges:
    weights = dict(g_dir.nodes(data='weight', default=np.inf))
    h_dir.add_edges_from(zip([str(i) + "'" for i in set(g_dir.nodes).difference(Y)],
                             [str(i) for i in set(g_dir.nodes).difference(Y)],
                             [{'weight': weights[i]} for i in set(g_dir.nodes).difference(Y)]))

    return [h_dir, h_bi, {str(i) for i in Y}]


# A function that transforms an edge ID instance to an intervention instance. The input are the Directed and Bidirected
# graphs with a set of costs on edges. The output are the corresponding Directed and Bidirected graphs with costs on
# nodes. Q[Y] is the desired query to identify.
def edgeIDtoIntervention(g_dir, g_bi, Y):
    nodes = [(str(i), {'weight': np.inf}) for i in g_dir.nodes]
    h_dir = nx.DiGraph()
    h_bi = nx.Graph()
    h_dir.add_nodes_from(nodes)
    h_bi.add_nodes_from(nodes)

    weights_dir = {(x1, x2): w for (x1, x2, w) in g_dir.edges(data='weight', default=np.inf)}
    weights_bi = {(x1, x2): w for (x1, x2, w) in g_bi.edges(data='weight', default=np.inf)}

    # first replace each directed edge with a node:
    for (x1, x2) in g_dir.edges:
        new_node = 'd_' + str(x1) + '_' + str(x2)
        h_dir.add_node(new_node, weight=weights_dir[(x1, x2)])
        h_dir.add_edge(str(x1), new_node, weight=np.inf)
        h_dir.add_edge(new_node, str(x2), weight=np.inf)
        h_bi.add_node(new_node, weight=weights_dir[(x1, x2)])
        h_bi.add_edge(str(x1), new_node, weight=np.inf)

    # Now replace each bidirected edge with a node:
    Y_sorted = list(nx.topological_sort(g_dir.subgraph(Y)))
    for (x1, x2) in g_bi.edges:
        new_node = 'b_' + str(x1) + '_' + str(x2)
        h_bi.add_node(new_node, weight=weights_bi[(x1, x2)])
        h_bi.add_edge(str(x1), new_node, weight=np.inf)
        h_bi.add_edge(str(x2), new_node, weight=np.inf)
        h_dir.add_node(new_node, weight=weights_bi[(x1, x2)])
        if x1 not in Y:
            h_dir.add_edge(new_node, str(x1), weight=np.inf)
        elif x2 not in Y:
            h_dir.add_edge(new_node, str(x2), weight=np.inf)
        else:  # if both endpoints are in Y, we need more edges
            for x in set(g_dir.nodes).difference(Y):
                h_dir.add_edge(new_node, str(x), weight=np.inf)
    Y_hat = []  # we will make Q[Y_hat] identifiable in the new graph after transformation.
    # fix the extra subgraphs for making Y ancestral:
    for i in range(len(Y_sorted)):
        for j in range(i+1, len(Y_sorted)):
            y1 = Y_sorted[i]
            y2 = Y_sorted[j]
            subG_bi = nx.subgraph(g_bi, Y_sorted[i:j+1])
            exgEnd = 'exg_' + str(y1) + '_' + str(y2) + '_h_' + str(y2)
            h_dir.add_node(exgEnd, weight=np.inf)
            h_bi.add_node(exgEnd, weight=np.inf)
            for z in set(subG_bi.nodes).difference(y2):
                new_node = 'exg_' + str(y1) + '_' + str(y2) + '_h_' + str(z)
                h_dir.add_node(new_node, weight=np.inf)
                h_bi.add_node(new_node, weight=np.inf)
                h_dir.add_edge(new_node, exgEnd, weight=np.inf)
            for (z1, z2) in subG_bi.edges:
                new_node = 'exg_' + str(y1) + '_' + str(y2) + '_b_' + str(z1) + '_' + str(z2)
                child_nn = 'b_' + str(z1) + str(z2)
                bidir1_nn = 'exg_' + str(y1) + '_' + str(y2) + '_h_' + str(z1)
                bidir2_nn = 'exg_' + str(y1) + '_' + str(y2) + '_h_' + str(z2)
                h_dir.add_node(new_node, weight=np.inf)
                h_bi.add_node(new_node, weight=np.inf)
                h_bi.add_edge(new_node, bidir1_nn, weight=np.inf)
                h_bi.add_edge(new_node, bidir2_nn, weight=np.inf)
                h_dir.add_edge(new_node, child_nn, weight=np.inf)
            exg1 = 'exg_' + str(y1) + '_' + str(y2) + '_h_' + str(y1)
            for z in Y_sorted[i:j+1]:
                h_dir.add_edge(str(z), exg1, weight=np.inf)
            h_bi.add_edge(str(y2), exg1)
            Y_hat.append(exgEnd)

    return [h_dir, h_bi, set(Y_hat).union(Y_sorted)]


# a function to solve the minimum hitting set problem.
# exact approach is a brute-force to check every possible combination,
# approx approach is the greedy algorithm choosing the node maximising the marginal gain at each iteration.
# The universe is U, T is the list of sets.
# costs is a dict mapping each member of the universe to its corresponding cost.
def solveHittingSet(U, T, costs, exact=False):
    if exact:
        setUnion = set.union(*(t for t in T))
        sorted_members, sorted_costs = (np.array(t) for t in zip(*sorted({m: costs[m] for m in setUnion}.items(),
                                                                         key=lambda item: item[1])))
        minCost = np.inf
        for i in range(0, len(setUnion) + 1):  # all possible subsets
            for sub in itertools.combinations(range(len(setUnion)), i):
                subset = set(sorted_members[list(sub)])
                subCost = np.sum(sorted_costs[list(sub)])
                if subCost < minCost:
                    doesHit = True
                    for t in T:
                        if len(subset.intersection(t)) == 0:
                            doesHit = False
                            break
                    if doesHit:
                        hittingSet = subset
                        minCost = subCost
            if minCost < sum(sorted_costs[:i + 1]):
                break
    else:  # Greedy Approximation Algorithm
        if len(T) == 1:
            costsT = {t: costs[t] for t in T[0]}
            return {min(costsT, key=costsT.get)}
        hittingSet = []
        appearances = {u: [] for u in U}
        for i in range(len(T)):
            seti = T[i]
            for x in seti:
                appearances[x].append(i)
        num_sets = len(T)
        while num_sets > 0:
            appearanceNumber = {u: len(appearances[u])/costs[u] for u in U}
            xtoAdd = max(appearanceNumber, key=appearanceNumber.get)
            hittingSet.append(xtoAdd)
            num_sets -= len(appearances[xtoAdd])
            for i in copy(appearances[xtoAdd]):
                seti = T[i]
                for y in seti:
                    appearances[y].remove(i)
    return hittingSet


# Solve minimum-cost intervention through Algorithm 2 (min-cost paper)
# input arguments: g is an admg with weights on nodes, S is the query (as in Q[S])
# third argument decides whether we want to solve it exactly or with a greedy algorithm with logarithmic factor approx.
# returns the optimal intervention set, its cost, and the number of hedges discovered throughout the algorithm.
def Alg2(g, S, hittingSetExactSolver=True):  # Exact (or Approx.) Algorithm
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)  # not to change anything in g
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:  # identifiability already achieved.
        return [dirP, g.interventionCost(dirP), 0]
    H_init = {tuple(c): h.hedgeHull(c) for c in unid_comps}  # hedge hulls of each component
    H_hitset = copy(H_init)
    H = copy(H_hitset)
    H_uni = set.union(*H.values())
    hedgesList = []  # will keep track of all the discovered hedges.
    appearances = {u: [c for c in unid_comps if u in H[tuple(c)]] for u in H_uni.difference(S)}  # which node
    # of H appears in the hedge hull of which nodes of S
    identified_comp = {tuple(c): False for c in unid_comps}
    while True:
        tempInterventionSet = []
        # hedge discovery begins:
        H = copy(H_hitset)
        while True:  # At the end of this while loop, we have an intervention set (temp) which makes Q[S] identifiable.
            while True:  # intervene on one variable greedily until Q[S] becomes identifiable.
                greedyGain = {u: len(appearances[u]) / g.interventionCost({u}) for u in appearances.keys()}  # gain of
                # intervention on a node of H
                xtoIntervene = max(greedyGain, key=greedyGain.get)
                for c in appearances[xtoIntervene]:
                    if not identified_comp[tuple(c)]:
                        temp_hhull = h.hedgeHull(c, H[tuple(c)].difference([xtoIntervene]))
                        for v in H[tuple(c)].difference(S.union(temp_hhull)):
                            appearances[v].remove(c)
                        if temp_hhull == c:
                            identified_comp[tuple(c)] = True  # Q[c] is identified
                            tempInterventionSet.append(xtoIntervene)
                            hedgesList.append(H[tuple(c)].difference(S))   # smallest discovered hedge for Q[c]
                        else:
                            H[tuple(c)] = temp_hhull
                if all(identified_comp.values()):
                    break
            H = {tuple(c): h.hedgeHull(c, H_hitset[tuple(c)].difference(tempInterventionSet)) for c in unid_comps}
            identified_comp = {tuple(c): H[tuple(c)] == c for c in unid_comps}
            if all(identified_comp.values()):
                break
            appearances = {u: [c for c in unid_comps if u in H[tuple(c)]] for u in H_uni.difference(S)}
        # a round of hedge discovery is done. we solve the minimum hitting set for all the hedges already found.
        hittingSetSolution = solveHittingSet(list(h.get_nodes()), hedgesList, h.get_nodeCosts(),
                                             exact=hittingSetExactSolver)
        H_hitset = {tuple(c): h.hedgeHull(c, H_init[tuple(c)].difference(hittingSetSolution)) for c in unid_comps}
        identified_comp = {tuple(c): H_hitset[tuple(c)] == c for c in unid_comps}
        if all(identified_comp.values()):  # Problem solved, all hedges hit
            break
        appearances = {u: [c for c in unid_comps if u in H_hitset[tuple(c)]] for u in H_uni.difference(S)}
    I = dirP.union(hittingSetSolution)
    return [I, g.interventionCost(I), len(hedgesList)]

# minimum vertex cut between two sets of variables. Solves both on directed and bidirected graphs.
# g is a networkx graph (either Graph or Digraph),
# source and target are a set of nodes of g.
# returns a list of min-vertex-cut between the source and target.
# IMPORTANT NOTE: Source nodes CAN be included in the cut, unless included in forbidden!
# this is for our purposes of min-cost intervention, where intervention on source nodes is possible (but not
# target nodes, which will be included in forbidden.)
# forbidden is a set of forbidden nodes (including, but not limited to target nodes)
def solveMinVertexCut(g, source, target, forbidden):
    directed = nx.is_directed(g)
    weights = dict(g.nodes(data='weight', default=np.inf))
    # make sure we do not include S nodes in the min-cut:
    for f in forbidden:
        weights[f] = np.inf
    # transform vertex-cut to edge-cut:
    h = nx.DiGraph()
    for v in g.nodes:
        h.add_edge(str(v) + "/1", str(v) + "/2", capacity=weights[v])
        for w in g.adj[v]:  # successors of v in g:
            h.add_edge(str(v) + "/2", str(w) + "/1", capacity=np.inf)
    # add a node and connect it to the source nodes:
    for v in source:
        h.add_edge("x_source", str(v) + "/1", capacity=np.inf)
        if not directed:
            h.add_edge(str(v) + "/2", "x_source", capacity=np.inf)
    # add a node and connect all target nodes to it:
    for t in target:
        h.add_edge(str(t) + "/2", "y_target", capacity=np.inf)
        if not directed:
            h.add_edge("y_target", str(t) + "/1", capacity=np.inf)

    # the graph is constructed. solve the min-cut:
    _, partition = nx.minimum_cut(h, "x_source", "y_target")
    reachable, non_reachable = partition

    # take the smaller of the reachable and non_reachable:
    if len(reachable) < len(non_reachable):
        part = list(reachable)
        part.remove("x_source")
    else:
        part = list(non_reachable)
        part.remove("y_target")

    # find the edges in the cut (representing the nodes)
    node_list = [v.split('/')[0] for v in part]
    return [v for v in node_list if node_list.count(v) == 1]


# a heuristic post-process to reduce the cost of heuristic algorithms:
# g is an ADMG, S is the query as in Q[S],
# A is a list containing the output of a heuristic alg.
def heuristicPostProcess(g, S, A):      # make A smaller as long as Q[S] is identifiable
    weights = g.get_nodeCosts()
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    A, _ = (list(x) for x in zip(*sorted({a: weights[a] for a in A}.items(), key=lambda item: item[1])))
    V = list(set(g.nodes).difference(A))
    for a in A:
        if all(g.isIdentifiable(c, set(V+[a])) for c in comps):
            V += [a]
    return set(g.nodes).difference(V)


# Min-cut based heuristic algorithm for minimum-cost intervention
# receives an instance of ADMG with node weights along with the query Q[S]
# returns a set which is sufficient to intervene upon for identification of Q[S], along with its cost.
# followed by an optional heuristic post-process to reduce the cost
def heuristicMinCut2(g, S, postProcess=True):     # break the ancestral sets where bidir(S) are present.
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:    # dirP is enough for the identification of Q[S]
        return [dirP, g.interventionCost(dirP)]
    S_unid = set.union(*unid_comps)
    H = set.union(*[h.hedgeHull(c) for c in unid_comps])
    # Construct the graph \mathcal{H}:
    dirSubg = h.nodeSubgraph(H, directed=True)
    bidirS = h.bidir(S_unid, H)
    try:
        minCut = minCut = solveMinVertexCut(dirSubg, bidirS, S_unid, S)
    except nx.NetworkXUnbounded:
        print('could not find a solution. returning infinite cost.')
        return [[], np.inf]
    if postProcess:
        minCut = heuristicPostProcess(h, S_unid, minCut)
    I = dirP.union(minCut)
    return [I, g.interventionCost(I)]



# Min-cut based heuristic algorithm for minimum-cost intervention
# receives an instance of ADMG with node weights along with the query Q[S]
# returns a set which is sufficient to intervene upon for identification of Q[S], along with its cost.
# followed by an optional heuristic post-process to reduce the cost
def heuristicMinCut1(g, S, postProcess=True):     # break the c-components where pa(S) are present.
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:  # dirP is enough for the identification of Q[S]
        return [dirP, g.interventionCost(dirP)]
    S_unid = set.union(*unid_comps)
    H = set.union(*[h.hedgeHull(c) for c in unid_comps])
    # Construct the graph \mathcal{H}:
    biSubg = h.nodeSubgraph(H, directed=False)  # bidirected subgraph
    parentS = h.parents(S_unid, H)
    try:
        minCut = solveMinVertexCut(biSubg, parentS, S_unid, S)
    except nx.NetworkXUnbounded:
        print('could not find a solution. returning infinite cost.')
        return [[], np.inf]
    if postProcess:
        minCut = heuristicPostProcess(h, S_unid, minCut)
    I = dirP.union(minCut)
    return [I, g.interventionCost(I)]


# Naive greedy algorithm for minimum cost intervention.
# intervenes upon nodes greedily until Q[S] becomes identifiable.
# receives an ADMG g, query Q[S], and is followed by an optional post-process to reduce the cost.
def naiveGreedy(g, S, postProcess=True):  # Reduce the sum of the cost of the remaining hedge hull
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:  # identifiability already achieved.
        return [dirP, g.interventionCost(dirP)]
    H = {tuple(c): h.hedgeHull(c) for c in unid_comps}  # hedge hulls of each component
    H_uni = set.union(*H.values())
    appearances = {u: [c for c in unid_comps if u in H[tuple(c)]] for u in H_uni.difference(S)}  # which node
    I = []
    identified_comp = {tuple(c): False for c in unid_comps}
    while True:
        greedyGain = {u: len(appearances[u]) / g.interventionCost({u}) for u in appearances.keys()}  # gain of
        xtoIntervene = max(greedyGain, key=greedyGain.get)
        I.append(xtoIntervene)
        for c in appearances[xtoIntervene]:
            if not identified_comp[tuple(c)]:
                new_hh = h.hedgeHull(c, H[tuple(c)].difference([xtoIntervene]))
                for v in H[tuple(c)].difference(new_hh).difference(S):
                    appearances[v].remove(c)
                if new_hh == c:
                    identified_comp[tuple(c)] = True
                else:
                    H[tuple(c)] = new_hh
        if all(identified_comp.values()):
            break
    if postProcess:
        I = heuristicPostProcess(h, S, I)
    I = dirP.union(I)
    return [I, g.interventionCost(I)]



# Min-cut based heuristic algorithm for edge ID
# receives an instance of ADMG with edge weights along with the query Q[{y}]
# returns a set of edges which is sufficient to delete for identification of Q[{y}], along with its cost.
def heuristicEdgeId2(g, y):     # break the ancestral sets where bidir(y) are present.
    y = set(y)
    if len(y) > 1:
        warnings.warn('Call to heuristicEdgeId2: this algorithm only works for single outcome!')
        return
    H = g.hedgeHull(y)
    if H == y:
        return [[], 0]
    bidirY = g.bidir(y, H)
    h_dir = g.nodeSubgraph(H, directed=True)
    h_bi = g.nodeSubgraph(bidirY.union(y), directed=False)
    h_mincut = nx.DiGraph()
    for (v, w, c) in h_dir.edges(data='weight', default=np.inf):
        h_mincut.add_edge(v, w, capacity=c)
    h_mincut.add_node('source')
    for (_, v, c) in h_bi.edges(list(y)[0], data='weight', default=np.inf):
        h_mincut.add_edge('source', v, capacity=c)

    try:
        cut_value, partition = nx.minimum_cut(h_mincut, 'source', list(y)[0])
    except nx.NetworkXUnbounded:
        print('could not find a solution. returning infinite cost.')
        return [[], np.inf]
    reachable, non_reachable = partition
    cutset = []
    for u, nbrs in ((n, h_mincut[n]) for n in reachable):
        if u == 'source':
            for v in nbrs:
                if v in non_reachable:
                    cutset.append((v, list(y)[0], 'bi'))
        else:
            for v in nbrs:
                if v in non_reachable:
                    cutset.append((u, v, 'dir'))
    return [cutset, cut_value]


# Min-cut based heuristic algorithm for edge ID 1
# receives an instance of ADMG with edge weights along with the query Q[{y}]
# returns a set of edges which is sufficient to delete for identification of Q[{y}], along with its cost.
def heuristicEdgeId1(g, y):     # break the c-components where parents(y) are present.
    y = set(y)
    if len(y) > 1:
        warnings.warn('Call to heuristicEdgeId1: this algorithm only works for single outcome!')
        return
    H = g.hedgeHull(y)
    if H == y:
        return [[], 0]
    parentsY = g.parents(y, H)
    h_bi = g.nodeSubgraph(H, directed=False)
    h_dir = g.nodeSubgraph(parentsY.union(y), directed=True)
    h_mincut = nx.DiGraph()
    for (v, w, c) in h_bi.edges(data='weight', default=np.inf):
        h_mincut.add_edge(v, w, capacity=c)
        h_mincut.add_edge(w, v, capacity=c)
    h_mincut.add_node('source')
    for (v, _, c) in h_dir.in_edges(list(y)[0], data='weight', default=np.inf):
        h_mincut.add_edge('source', v, capacity=c)
        h_mincut.add_edge(v, 'source', capacity=c)

    try:
        cut_value, partition = nx.minimum_cut(h_mincut, 'source', list(y)[0])
    except nx.NetworkXUnbounded:
        print('could not find a solution. returning infinite cost.')
        return [[], np.inf]
    reachable, non_reachable = partition
    cutset = []
    for u, nbrs in ((n, h_mincut[n]) for n in reachable):
        if u == 'source':
            for v in nbrs:
                if v in non_reachable:
                    cutset.append((v, list(y)[0], 'dir'))
        else:
            for v in nbrs:
                if v in non_reachable:
                    cutset.append((u, v, 'bi'))
    return [cutset, cut_value]



def hedgeHull2(digraph, bigraph, S, H=None):
    if H is None:
        H = set(digraph.nodes)  # The whole graph
    if warningsOn:
        if not S.issubset(H):
            warnings.warn("Call to hedgeHull: S is not a subset of H!")
            return None
        if not nx.is_connected(nx.subgraph(bigraph, S)):
            warnings.warn("Call to hedgeHull: S is not a c-component!")
            return None
    subset = copy(H)
    # Ancestor of S in H:
    anc_set = ancestors_general(nx.subgraph(digraph, subset), S)
    s = list(S)[0]
    # connected component of S in anc_set:
    con_comp = nx.node_connected_component(nx.subgraph(bigraph, anc_set), s)
    if con_comp == subset:
        return subset
    subset = con_comp
    # Find the largest set of nodes which is ancestral for S and is a c-component:
    while True:
        anc_set = ancestors_general(nx.subgraph(digraph, subset), S)
        if anc_set == subset:
            return subset
        subset = anc_set
        con_comp = nx.node_connected_component(nx.subgraph(bigraph, subset), s)
        if con_comp == subset:
            return subset
        subset = con_comp

def isIdentifiable2(digraph, bigraph, S, H=None):
    if H is None:
        H = digraph.nodes  # The whole graph
    if warningsOn:
        if not set(S).issubset(H):
            # S is not a subset of H, so not ID
            return False
    if set(S) == hedgeHull2(digraph=digraph, bigraph=bigraph, S=S, H=H):
        return True
    return False


# brute force algorithm for edge ID:
# g is an admg, Y is the set in query Q[Y]
def edgeIDbrute(digraph, bigraph, Y):
    MH = hedgeHull2(digraph=digraph, bigraph=bigraph, S=Y)
    check = isIdentifiable2(digraph=digraph, bigraph=bigraph, S=Y, H=MH)
    E = set()
    if check == True:  # if already identifiable, return cost 0
        return E, 0

    else:
        # get maximal hedge subgraphs
        multigraph_to_digraph_graph
        hedge_digraph = nx.subgraph(digraph, MH)
        hedge_bigraph = nx.subgraph(bigraph, MH)
        bi_weights = np.array([weight[2] for weight in list(hedge_bigraph.edges.data('weight'))])
        bi_edges = np.array([weight[:2] for weight in list(hedge_bigraph.edges.data('weight'))])
        di_weights = np.array([weight[2] for weight in list(hedge_digraph.edges.data('weight'))])
        di_edges = np.array([weight[:2] for weight in list(hedge_digraph.edges.data('weight'))])

        all_edges = np.concatenate([bi_edges, di_edges])
        all_weights = np.concatenate([bi_weights, di_weights])

        all_weights_sorted = all_weights[np.argsort(all_weights)]
        all_edges_sorted = all_edges[np.argsort(all_weights)]
        all_edge_names = np.arange(0, len(all_edges_sorted))

        best_cost = np.inf
        j = 1  # size of combinations of edges
        while True:
            combs = list(itertools.combinations(np.arange(0, len(all_edge_names)), j))
            if len(combs) != 0:
                k = 0
                while True:  # iterate through combinations
                    comb = list(combs[k])
                    edges = all_edge_names[comb]   # get edges names in combination

                    # check if (e is empty) OR (is not empty AND is not subset of current edge combination)
                    if (len(E) == 0) or ((len(E) != 0) and (E.issubset(edges) == False)):
                        edges_full = all_edges_sorted[edges]
                        temp_bi = hedge_bigraph.copy()
                        temp_di = hedge_digraph.copy()
                        # remove proposed edges from bi and di hedge graphs
                        for edge in edges_full:
                            try:
                                temp_bi.remove_edge(*edge)
                            except:
                                pass
                            try:
                                temp_di.remove_edge(*edge)
                            except:
                                pass
                        # check if ID:
                        if isIdentifiable2(digraph=temp_di, bigraph=temp_bi, S=Y, H=MH):
                            costs = all_weights_sorted[list(comb)]  # get costs in combination
                            current_cost = costs.sum()  # get cost of edge intervention

                            if current_cost < best_cost:  # if solution is an improvement over previous solution
                                best_cost = current_cost  # update cost
                                E = set(edges)  # store solution

                                if best_cost < max(
                                        all_weights_sorted):  # if the best cost is less than the max costs of any edge, do some pruning
                                    # prune_ind = np.where((all_weights_sorted < best_cost) == False)[0][0]  # get ind for pruning expensive edges
                                    # all_edge_names = all_edge_names[:prune_ind]  # prune edges
                                    #
                                    # # # prune list of combinations if the combination contains any edges greater than the prune_ind
                                    # to_removes = [comb for comb in combs if (np.array(comb) >= prune_ind).any()]
                                    # for to_remove in to_removes: combs.remove(to_remove)

                                    # if the best cost is less than the max costs of any combination of edges, do some pruning
                                    to_removes = []
                                    for comb in combs:
                                        cost = all_weights_sorted[list(comb)].sum()
                                        if cost > best_cost:
                                            to_removes.append(comb)
                                    for to_remove in to_removes: combs.remove(to_remove)

                    k += 1
                    if k >= len(combs):  # if we have gotten to the end of the (possibly pruned) list of combinations, break
                        break

            j += 1
            if j >= len(all_edge_names):  # if we have run out of edges in the (possibly pruned) list, break
                break

    return all_edges_sorted[list(E)], best_cost



def edgeIDbrutev2(digraph, bigraph, Y, upper_bound=np.inf):
    MH = hedgeHull2(digraph=digraph, bigraph=bigraph, S=Y)
    check = isIdentifiable2(digraph=digraph, bigraph=bigraph, S=Y, H=MH)
    E = set()
    if check == True:  # if already identifiable, return cost 0
        return E, 0

    else:
        # get maximal hedge subgraphs
        multigraph_to_digraph_graph
        hedge_digraph = nx.subgraph(digraph, MH)
        hedge_bigraph = nx.subgraph(bigraph, MH)
        bi_weights = np.array([weight[2] for weight in list(hedge_bigraph.edges.data('weight'))])
        bi_edges = np.array([weight[:2] for weight in list(hedge_bigraph.edges.data('weight'))])
        di_weights = np.array([weight[2] for weight in list(hedge_digraph.edges.data('weight'))])
        di_edges = np.array([weight[:2] for weight in list(hedge_digraph.edges.data('weight'))])

        all_edges = np.concatenate([bi_edges, di_edges])
        all_weights = np.concatenate([bi_weights, di_weights])

        all_weights_sorted = all_weights[np.argsort(all_weights)]
        all_edges_sorted = all_edges[np.argsort(all_weights)]
        all_edge_names = np.arange(0, len(all_edges_sorted))

        best_cost = upper_bound
        j = 1  # initial size of combinations of edges
        iter_num = 0
        while True:
            combs = list(itertools.combinations(np.arange(0, len(all_edge_names)), j))


            if (iter_num == 0) and (upper_bound != np.inf):
                print('Using cost upper bound of: ', upper_bound)
                # if the first run and we don't have infinite cost limit, see if we can prune the search space
                to_removes = []
                for comb in combs:
                    cost = all_weights_sorted[list(comb)].sum()
                    if cost > best_cost:
                        to_removes.append(comb)
                for to_remove in to_removes: combs.remove(to_remove)

            if len(combs) != 0:
                k = 0
                while True:  # iterate through combinations
                    comb = list(combs[k])
                    edges = all_edge_names[comb]   # get edges names in combination

                    # check if (e is empty) OR (is not empty AND is not subset of current edge combination)
                    if (len(E) == 0) or ((len(E) != 0) and (E.issubset(edges) == False)):
                        edges_full = all_edges_sorted[edges]
                        temp_bi = hedge_bigraph.copy()
                        temp_di = hedge_digraph.copy()
                        # remove proposed edges from bi and di hedge graphs
                        for edge in edges_full:
                            try:
                                temp_bi.remove_edge(*edge)
                            except:
                                pass
                            try:
                                temp_di.remove_edge(*edge)
                            except:
                                pass
                        # check if ID:
                        if isIdentifiable2(digraph=temp_di, bigraph=temp_bi, S=Y, H=MH):
                            costs = all_weights_sorted[list(comb)]  # get costs in combination
                            current_cost = costs.sum()  # get cost of edge intervention

                            if current_cost < best_cost:  # if solution is an improvement over previous solution
                                best_cost = current_cost  # update cost
                                E = set(edges)  # store solution

                                if best_cost < max(
                                        all_weights_sorted):  # if the best cost is less than the max costs of any combination of edges, do some pruning
                                    to_removes = []
                                    for comb in combs:
                                        cost = all_weights_sorted[list(comb)].sum()
                                        if cost > best_cost:
                                            to_removes.append(comb)
                                    for to_remove in to_removes: combs.remove(to_remove)

                    k += 1
                    if k >= len(combs):  # if we have gotten to the end of the (possibly pruned) list of combinations, break
                        break

            j += 1
            if j >= len(all_edge_names):  # if we have run out of edges in the (possibly pruned) list, break
                break
            iter_num += 1

    return all_edges_sorted[list(E)], best_cost



def edgeIDbrutev3(digraph, bigraph, Y, upper_bound=np.inf):
    MH = hedgeHull2(digraph=digraph, bigraph=bigraph, S=Y)
    check = isIdentifiable2(digraph=digraph, bigraph=bigraph, S=Y, H=MH)
    E = set()
    if check == True:  # if already identifiable, return cost 0
        return E, 0

    else:
        # get maximal hedge subgraphs
        multigraph_to_digraph_graph
        hedge_digraph = nx.subgraph(digraph, MH)
        hedge_bigraph = nx.subgraph(bigraph, MH)
        bi_weights = np.array([weight[2] for weight in list(hedge_bigraph.edges.data('weight'))])
        bi_edges = np.array([weight[:2] for weight in list(hedge_bigraph.edges.data('weight'))])
        di_weights = np.array([weight[2] for weight in list(hedge_digraph.edges.data('weight'))])
        di_edges = np.array([weight[:2] for weight in list(hedge_digraph.edges.data('weight'))])

        all_edges = np.concatenate([bi_edges, di_edges])
        all_weights = np.concatenate([bi_weights, di_weights])

        all_weights_sorted = all_weights[np.argsort(all_weights)]
        all_edges_sorted = all_edges[np.argsort(all_weights)]
        all_edge_names = np.arange(0, len(all_edges_sorted))

        best_cost = upper_bound
        j = 1  # initial size of combinations of edges
        iter_num = 0
        while True:
            # get all j=lengthcombinations of edges
            combs = list(itertools.combinations(np.arange(0, len(all_edge_names)), j))
            # get all costs for these combinations
            combs_costs = []
            for comb in combs:
                cost = all_weights_sorted[list(comb)].sum()
                combs_costs.append(cost)
            # sort costs and combinations by cost
            order_inds = np.argsort(combs_costs)
            combs_costs = np.asarray(combs_costs)[order_inds]
            combs = np.asarray(combs)[order_inds]

            to_removes_inds = []
            for jj, comb in enumerate(combs):
                cost = combs_costs[jj]
                if cost > best_cost:
                    to_removes_inds.append(jj)

            if len(to_removes_inds) > 0:
                print('pruning')
                combs = np.delete(combs, to_removes_inds, axis=0).reshape(-1, j)
                combs_costs = np.delete(combs_costs, to_removes_inds)

            if len(combs) != 0:
                k = 0
                while True:  # iterate through combinations
                    if k >= len(combs):  # if we have gotten to the end of the (possibly pruned) list of combinations, break
                        break

                    comb = list(combs[k])
                    current_cost = combs_costs[k]
                    edges = all_edge_names[comb]   # get edges names in combination

                    # check if (e is empty) OR (is not empty AND is not subset of current edge combination)
                    if (len(E) == 0) or ((len(E) != 0) and (E.issubset(edges) == False)):
                        edges_full = all_edges_sorted[edges]
                        temp_bi = hedge_bigraph.copy()
                        temp_di = hedge_digraph.copy()
                        # remove proposed edges from bi and di hedge graphs
                        for edge in edges_full:
                            try:
                                temp_bi.remove_edge(*edge)
                            except:
                                pass
                            try:
                                temp_di.remove_edge(*edge)
                            except:
                                pass
                        # check if ID:
                        if isIdentifiable2(digraph=temp_di, bigraph=temp_bi, S=Y, H=MH):
                            if current_cost < best_cost:  # if solution is an improvement over previous solution
                                best_cost = current_cost  # update cost
                                E = set(edges)  # store solution

                    k += 1

            j += 1
            if j >= len(all_edge_names):  # if we have run out of edges in the (possibly pruned) list, break
                break
            iter_num += 1

    return all_edges_sorted[list(E)], best_cost


# if we solve edge ID using the conversion to min-cost intervention, we need to map nodes into edge names.
def mapNodetoEdge(nodes):
    edges = []
    for node in nodes:
        edge_type, x1, x2 = node.split('_')
        if edge_type == 'b':
            edges.append((x1, x2, 'bi'))
        else:
            edges.append((x1, x2, 'dir'))
    return edges


# if we solve min-cost intervention using the conversion to edge ID, we need to map edges into node names.
def mapEdgetoNode(edges):
    nodes = []
    for x1, _, _ in edges:
        nodes.append(x1.split("'")[0])
    return nodes


if __name__ == '__main__':

    # build a DiGraph and a Graph:
    g_dir = nx.DiGraph()
    g_bi = nx.Graph()
    nodes = [('a', {'weight': 5}), ('2', {'weight': 4}), ('3', {'weight': 3}), ('4', {'weight': 5}),
             ('5', {'weight': 1})]
    g_dir.add_nodes_from(nodes)
    g_bi.add_nodes_from(nodes)
    dir_edges = [('a', '2', {'weight': 1}), ('3', '2', {'weight': 0.01}), ('4', 'a', {'weight': 2}),
                 ('5', '3', {'weight': 3}), ('4', '5', {'weight': 3})]
    bi_edges = [('a', '4', {'weight': 2}), ('5', '4', {'weight': 1}), ('4', '2', {'weight': 2}),
                ('4', '3', {'weight': 1})]
    g_dir.add_edges_from(dir_edges)
    g_bi.add_edges_from(bi_edges)

    # build an instance of ADMG with g_dir and g_bi:
    g = ADMG(g_dir, g_bi)
    Y = {'2', '3'}

    # First approach: directly solve the edge ID problem, without conversion to min-cost intervention:
    print('Direct Approaches:')
    print('brute force edge ID: ' + str(edgeIDbrute(g, Y)))    # not implemented yet
    print('heuristic edge ID2: ' + str(heuristicEdgeId2(g, Y)))   # only works for single Y
    print('heuristic edge ID1: ' + str(heuristicEdgeId1(g, Y)))  # only works for single Y

    # Second approach: convert it to the min-cost intervention and solve it
    h_dir, h_bi, y_hat = edgeIDtoIntervention(g_dir, g_bi, Y)
    h = ADMG(h_dir, h_bi)
    # h.plotWithNodeWeights()
    print()
    print('Transformation-based algorithms:')
    intervention, cost = heuristicMinCut2(h, y_hat)       # heuristic min-cut based algorithm 2 for min-cost
    # intervention
    print('heuristicMinCut2: [' + str(mapNodetoEdge(intervention)) + ', ' + str(cost) + ']')
    intervention, cost = heuristicMinCut1(h, y_hat)        # heuristic min-cut based algorithm 1 for min-cost
    # intervention
    print('heuristicMinCut1: [' + str(mapNodetoEdge(intervention)) + ', ' + str(cost) + ']')
    intervention, cost = naiveGreedy(h, y_hat, postProcess=True)            # a naive greedy for min-cost intervention
    print('Naive Greedy: [' + str(mapNodetoEdge(intervention)) + ', ' + str(cost) + ']')
    intervention, cost = h.optimalIntervention(y_hat)  # brute-force algorithm for min-cost intervention which checks
    # for every subset
    print('Brute force: [' + str(mapNodetoEdge(intervention)) + ', ' + str(cost) + ']')
    intervention, cost, _ = Alg2(h, y_hat)              # minimum-hitting set based more efficient algorithm for
    # min-cost intervention
    print('Exact Alg2: [' + str(mapNodetoEdge(intervention)) + ', ' + str(cost) + ']')
    intervention, cost, _ = Alg2(h, y_hat, hittingSetExactSolver=False)     # same algorithm, but solves the
    # min-hitting set approximately
    print('Approx. Alg2: [' + str(mapNodetoEdge(intervention)) + ', ' + str(cost) + ']')

    # Third approach: double transformation for multiple outcomes: do we need it?
    # Important Note: with the current transformation of interventionToEdgeID, heuristicEdgeId1 cannot find any answers
    # g_dir2, g_bi2, y_tilde = interventionToedgeID(h_dir, h_bi, y_hat)
    # g2 = ADMG(g_dir2, g_bi2)
    # edges_ID, cost = heuristicEdgeId2(g2, y_tilde)
    # print()
    # print('double transform, heuristic edge ID 2: [' + str(mapNodetoEdge(mapEdgetoNode(edges_ID))) +
    #       ', ' + str(cost) + ']')
