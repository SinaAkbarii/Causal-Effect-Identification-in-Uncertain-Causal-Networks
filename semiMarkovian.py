from random import sample
from copy import copy
import numpy as np
import warnings
import itertools
from matplotlib import pyplot as plt
import networkx as nx
from math import comb


class SemiMarkovian:

    # init
    def __init__(self, n, adjDirected=None, adjBidirected=None, cost=None, seed=None,
                 pDir=None, pBidir=None, costRandom=True, hedgeDensity=0.01, layers=1, S=None):
        self.n = n  # number of nodes
        if seed is None:
            if adjDirected is None:
                adjDirected = np.full([n, n], False, dtype=bool)
            if adjBidirected is None:
                adjBidirected = np.full([n, n], False, dtype=bool)
        elif seed == 'erdos':   # Build one directed and one undirected Erdos-Renyi Graphs
            if pDir is None:
                pDir = 0.5
            if pBidir is None:
                pBidir = 0.5
            if adjDirected is None:
                adjDirected = np.triu(np.random.random([n, n]), k=1) > (1-pDir)
            if adjBidirected is None:
                adjBidirected = np.triu(np.random.random([n, n]), k=1) > (1-pBidir)
                adjBidirected = np.bitwise_or(adjBidirected, np.transpose(adjBidirected))
        elif seed == 'hedgelayer':
            adjDirected = np.full([n, n], False, dtype=bool)
            adjBidirected = np.full([n, n], False, dtype=bool)
            if S is None:
                S = [n-1]
            GminusS = list(range(min(S)))
            for k in np.add(range(layers), 2):
                numhedges = max(int(np.round(comb(len(GminusS), k) * hedgeDensity)), 1)
                for i in range(numhedges):
                    subset = sorted(list(sample(GminusS, k)))
                    if adjDirected[subset[0], S[0]]:
                        continue
                    hedgeSet = subset+S
                    numNotHandled = copy(k)
                    while numNotHandled > 0:  # Add directed edges so that the set K\cup S becomes ancestral:
                        numToHandle = np.random.randint(1, min(k, numNotHandled+1))
                        for j in np.add(range(numToHandle), numNotHandled-numToHandle):
                            adjDirected[hedgeSet[j], hedgeSet[numNotHandled]] = True
                        numNotHandled -= numToHandle
                    # Add bidirected edges so that the set K\cup S becomes c-connected:
                    adjBidirected[S[0], subset[0]] = True
                    adjBidirected[subset[0], S[0]] = True
                    np.random.shuffle(subset)
                    for j in range(len(subset)-1):
                        adjBidirected[subset[j], subset[j+1]] = True
                        adjBidirected[subset[j+1], subset[j]] = True

        if cost is None:
            if costRandom:
                cost = np.random.randint(1, 9, size=n).astype(float)
            else:
                cost = np.full(n, 1, dtype=float)
        self.DAG = nx.DiGraph(adjDirected)
        self.latent = nx.Graph(adjBidirected)
        self.cost = cost

    # does a subgraph H form a hedge for S?
    def isHedge(self, S, H):
        if not set(S).issubset(set(H)):
            warnings.warn("Call to isHedge: S is not a subset of H!")
        if not nx.is_connected(nx.subgraph(self.latent, S)):
            warnings.warn("Call to isHedge: S is not a c-component!")
        # consider the subgraph over H:
        subLat_H = nx.subgraph(self.latent, H)
        if nx.is_connected(subLat_H):  # H is a c-component. Let's see if it is ancestral as well.
            subDAG_H = nx.subgraph(self.DAG, H)
            ancDAG = copy(S)
            for s in S:
                ancDAG += list(nx.ancestors(subDAG_H, s))
            if set(ancDAG) == set(H):  # this means H is ancestral.
                return True
        # print('S after isHedge = ' + str(S))
        return False

    # Construct the hedge hull of S in the subgraph H:
    def hedgeHull(self, S, H=None):
        if H is None:
            H = range(self.n)   # The whole graph
        if not set(S).issubset(set(H)):
            warnings.warn("Call to hedgeHull: S is not a subset of H!")
            return None
        subG = H    # We only consider the subgraph over subG
        singleCC = False    # subG is not a single c-component
        while True:
            # Start with the DAG (this order can be changed, i.e., start from the bidirected graph)
            ancestral = False
            dagG = nx.subgraph(self.DAG, subG)
            ancDAG = set(S)
            for s in S:
                ancDAG = ancDAG.union(nx.ancestors(dagG, s))
            if ancDAG == set(subG):
                if singleCC:    # subG is ancestral and single C-component, i.e., a Hedge
                    break
                ancestral = True
            subG = list(ancDAG)

            # Now check the latent Graph to see if subG is a c-component
            singleCC = False
            latG = nx.subgraph(self.latent, subG)
            latComponent = set()
            for s in S:
                if s not in latComponent:
                    latComponent = latComponent.union(nx.node_connected_component(latG, s))
            if latComponent == set(subG):
                # subG is ancestral and single C-component, i.e., a Hedge
                break
            singleCC = True
            subG = list(latComponent)

        return subG

    # Determine if S is identifiable in subgraph H
    def isIdentifiable(self, S, H=None):
        if H is None:
            H = range(self.n)   # The whole Graph
        if not set(S).issubset(set(H)):
            # S is not a subset of H, so not ID
            return False
        if set(S) == set(self.hedgeHull(S, H)):
            return True
        return False

    def interventionResult(self, S, I=[]):
        H = list(set(range(self.n)).difference(set(I)))
        interventionCost = np.sum(self.cost[I])
        return [self.isIdentifiable(S, H), interventionCost]

    def parents(self, S, H=None):
        if H is None:
            H = range(self.n)
        parS = []
        dagH = nx.subgraph(self.DAG, H)
        for s in S:
            parS += dagH.predecessors(s)
        parS = set(parS).difference(set(S))
        return list(parS)

    def bidir(self, S, H=None):
        if H is None:
            H = range(self.n)
        bidir1 = []
        latH = nx.subgraph(self.latent, H)
        for s in S:
            bidir1 += latH.neighbors(s)
        bidir1 = set(bidir1).difference(set(S))
        return list(bidir1)

    def directParents(self, S, H=None):     # directParents must be intervened upon.
        if H is None:
            H = range(self.n)
        return list(set(self.parents(S, H)).intersection(set(self.bidir(S, H))))

    def optimalIntervention(self, S, H=None):
        if H is None:
            H = range(self.n)
        dirParents = self.directParents(S, H)
        # dirParents must be intervened upon.
        baseCost = np.sum(self.cost[dirParents])
        if self.interventionResult(S, dirParents)[0]:    # if the set of direct Parents is enough to identify
            return [dirParents, baseCost]
        H = self.hedgeHull(S, list(set(H).difference(set(dirParents))))
        dirParents = set(dirParents)
        minCostAdd = np.inf
        optInterv = list(set(H).difference(set(S)))
        optCosts = sorted(copy(self.cost[optInterv]))
        for i in range(0, len(optInterv) + 1):  # all subsets
            for subset in itertools.combinations(optInterv, i):
                if np.sum(self.cost[list(subset)]) < minCostAdd:
                    I = list(set(subset).union(dirParents))
                    if self.interventionResult(S, I)[0]:
                        minCostAdd = np.sum(self.cost[list(subset)])
                        intervSet = I
            if minCostAdd < sum(optCosts[:i+1]):
                break
        return [intervSet, baseCost+minCostAdd]

    def smallestCostVertex(self, H):    # return the min cost vertex among H
        H = np.array(H)
        tempCost = np.full(self.n, np.inf, dtype=float)
        tempCost[H] = copy(self.cost[H])
        min_value = min(tempCost)
        return [list(tempCost).index(min_value), min_value]

    def permIntervene(self, I=None):     # return the graph after intervention on I
        if I is None:
            I = []
        adjD = copy(np.array(nx.adjacency_matrix(self.DAG).todense()))
        adjD[np.ix_(range(self.n), I)] = np.full([self.n, len(I)], False, dtype=bool)
        adjD[np.ix_(I, range(self.n))] = np.full([len(I), self.n], False, dtype=bool)
        adjB = copy(np.array(nx.adjacency_matrix(self.latent).todense()))
        adjB[np.ix_(range(self.n), I)] = np.full([self.n, len(I)], False, dtype=bool)
        adjB[np.ix_(I, range(self.n))] = np.full([len(I), self.n], False, dtype=bool)
        return [copy(self.n), adjD, adjB, copy(self.cost)]

    def countHedges(self, S, H=None):    # Count the number of hedges formed for S in H
        if H is None:
            H = range(self.n)
        count = 0
        if set(S).issubset(set(H)):
            H = self.hedgeHull(S, H)
            if set(H) == set(S):
                return 0
            HminusS = list(set(H).difference(set(S)))
            count += self.countHedges(S, list(set(H).difference([HminusS[0]])))
            for i in range(1, len(HminusS)):  # all subsets
                for subset in itertools.combinations(list(set(HminusS).difference([HminusS[0]])), i):
                    I = list(set(subset).union(S).union([HminusS[0]]))
                    if self.isHedge(S, I):
                        count += 1
        else:
            warnings.warn('S is not a subset of H!')
        return count

    # Change the cost of variables X from self.cost[X] to newCost. Note that X needs not be a single vertex
    def changeCost(self, X, newCost):
        self.cost[X] = newCost
        return

    def pruneS(self, S):    # delete outgoing edges of S
        for s in S:
            self.DAG.remove_edges_from(list(self.DAG.out_edges(s)))
        return

    def bidirectedEdgeSubgraph(self, H):
        if H is None:
            H = range(self.n)
        subAdj = np.full([self.n, self.n], False, dtype=bool)
        adjB = copy(np.array(nx.adjacency_matrix(self.latent).todense()))
        subAdj[np.ix_(H, H)] = adjB[np.ix_(H, H)]
        return subAdj

    def directedEdgeSubgraph(self, H):
        if H is None:
            H = range(self.n)
        subAdj = np.full([self.n, self.n], False, dtype=bool)
        adjD = copy(np.array(nx.adjacency_matrix(self.DAG).todense()))
        subAdj[np.ix_(H, H)] = adjD[np.ix_(H, H)]
        return subAdj

    def plotG(self, S=None, H=None):
        if S is None:
            S = []
        if H is None:
            H = range(self.n)   # The whole graph
        G1 = nx.DiGraph(self.directedEdgeSubgraph(H))
        G2 = nx.DiGraph(np.triu(self.bidirectedEdgeSubgraph(H)))
        pos = nx.spring_layout(G1)
        nx.draw_networkx_nodes(G1, pos, node_size=400, label=list(set(H).difference(set(S))),
                               nodelist=list(set(H).difference(set(S))),
                               node_color=None)
        nx.draw_networkx_nodes(G1, pos, node_size=400, label=S, nodelist=S, node_color='red')
        nx.draw_networkx_edges(G1, pos)
        nx.draw_networkx_edges(G2, pos, style=':', connectionstyle='arc3, rad=0.4', arrowsize=0.01, edge_color='grey')
        nx.draw_networkx_labels(G1, pos)
        for i in H:
            x, y = pos[i]
            plt.text(x+0.03, y+0.03, s=str(self.cost[i]), bbox=dict(facecolor='yellow', alpha=0.5),
                     horizontalalignment='center', fontsize=7)
        plt.show()
        return

    def sampleHedgeK(self, k, srate, S, H=None):    # how many hedges of size k are there? srate: sampling rate (0-1)
        if H is None:
            H = range(self.n)
        HminusS = list(set(H).difference(set(S)))
        num_hedge = 0
        if srate < 1:
            numtests = max(int(np.round(comb(len(HminusS), k)*srate)), 1)
            for i in range(numtests):
                subset = sample(HminusS, k)
                if self.isHedge(S, list(set(subset).union(set(S)))):
                    num_hedge += 1
        else:
            for subset in itertools.combinations(HminusS, k):
                if self.isHedge(S, list(set(subset).union(set(S)))):
                    num_hedge += 1
        return np.round(num_hedge / srate)

    def sampleHedge(self, srate, S, H=None):    # Counts the number of hedges formed for S in H. srate = sampling rate
        if H is None:
            H = range(self.n)
        H = self.hedgeHull(S, H)
        num_hedges = 0
        for k in np.add(range(len(H)-len(S)), 1):
            num_hedges += self.sampleHedgeK(k, srate, S, H)
        return num_hedges

    def sampleHedgeListK(self, k, srate, S, H=None):    # Generate a list of samples of the hedges of size K+|S|.
        if H is None:
            H = range(self.n)
        HminusS = list(set(H).difference(set(S)))
        numtests = max(int(np.round(comb(len(HminusS), k)*srate)), 1)
        hedgeList = []
        for i in range(numtests):
            subset = sample(HminusS, k)
            if self.isHedge(S, list(set(subset).union(set(S)))):
                hedgeList.append(tuple(sorted(subset)))
        return hedgeList

    def sampleHedgeList(self, srate, S, H=None):    # Generates the list of hedges. samples each subset with probability srate
        if H is None:
            H = range(self.n)
        H = self.hedgeHull(S, H)
        hedgeList = []
        for k in np.add(range(len(H)-len(S)), 1):
            hedgeList += self.sampleHedgeListK(k, srate, S, H)
        return hedgeList