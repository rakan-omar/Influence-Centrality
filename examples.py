from pyvis.network import Network
import networkx as nx
import numpy as np
import scipy as sp
import influence_centrality as ic
import random as rd



def bureaucracy(n, factor):
    G = nx.DiGraph()

    for i in range(n):
        G.add_node(i+1, value=1)
    for i in range(n):
        if (i < n - 1):
            G.add_weighted_edges_from([(i+1, i + 2, factor)])
        else:
            G.add_weighted_edges_from([(i+1, 1, factor)])
    return G


def democracy(n, factor = 0):
    G = nx.DiGraph()
    if factor == 0:
        factor = 1/n

    for i in range(n):
        G.add_node(i + 1, value = 1)

    for i in range(n):
        for j in range(n):
            if (i != j):
                G.add_weighted_edges_from([(i+1, j+1, factor)])
    
    return G

def hierarchy():
    G = nx.DiGraph()
    G.add_nodes_from([(1, {'value':0.9}), (2, {'value':0.8}), (3, {'value':0.8}), (4, {'value':0.7}), (5, {'value':0.7}), (6, {'value':0.7}), (7, {'value':0.6}), (8, {'value':0.6}),
                      (8, {'value':0.6}), (9, {'value':0.6})])

    G.add_weighted_edges_from([(1, 2, 0.5), (1, 3, 0.4), (2, 4, 0.4), (2, 5, 0.5), (3, 6, 0.6), (4, 7, 0.4), (5, 8, 0.3), (6, 9, 0.7)])
    
    return G
    

def path():
    G= nx.DiGraph()
  #  G.add_nodes_from([(1, {'value':1}), (2, {'value':1}), (3, {'value':1}),(4, {'value':1}), (5, {'value':1})])
   # G.add_weighted_edges_from([(1, 2, 0.7), (2, 3, 0.7), (3, 4, 0.7), (4, 5, 0.7)])
    G.add_nodes_from([(1, {'value':1}), (2, {'value':1}), (3, {'value':1}),(4, {'value':1}), (5, {'value':1})])
    G.add_weighted_edges_from([(1, 2, 0.7), (2, 3, 0.7), (3, 4, 0.7), (4, 5, 0.7)])
    return G


def trail():
    G = nx.DiGraph()
    G.add_nodes_from([(1, {'value':1}), (2, {'value':1}), (3, {'value':1}), (4, {'value':1}), (5, {'value':1}),(6, {'value':1})])
    G.add_weighted_edges_from([(1, 2, 0.7), (2, 3, 0.6), (3, 4, 0.5), (4, 1, 0.2), (6, 5, 0.5), (5, 1, 0.4)])
    return G

def complex_trail():
    G = nx.DiGraph()
    for i in range(20):
        G.add_node(i+1, value=rd.uniform(0, 1))

    G.add_weighted_edges_from([(1, 2, 0.15), (2, 12, 0.1), (3, 4, 0.2), (5, 8, 0.5), (1, 5, 0.2), (5, 6, 0.1), (6, 9, 0.4), (19, 20, 0.3),
                            (20, 2, 0.3), (4, 13, 0.2), (7, 14, 0.1), (15, 12, 0.3), (11, 10, 0.05), (10, 9, 0.1), (9, 12, 0.1), (9, 10, 0.1), (11, 18, 0.1),
                            (19, 18, 0.3), (18, 16, 0.2), (17, 20, 0.6), (17, 13, 0.1), (12, 18, 0.1), (9, 2, 0.1), (3, 5, 0.1), (2, 7, 0.1), (9, 10, 0.1)])
    return G

def draw_G(G, filename = 'network_graph'):
    #loop over all nodes, give size and labels
    for node in G.nodes:
        G.add_node(node, size=G.nodes[node]['value'] * 10, physics=False)
    
    graph = Network('700px', '700px', directed=True)
    graph.from_nx(G)
    for edge in graph.edges:
        edge['label'] = edge['weight']
    graph.show(filename + '.html')
    return graph

G = complex_trail()
G = ic.influence_centrality(G)

draw_G(G)


