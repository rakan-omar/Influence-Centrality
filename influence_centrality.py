from pyvis.network import Network
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as pyplot
import itertools as it
import csv


def weighted_sum_indegree(A, v, G): # (adjacency matrix, node)
    indeg = 0
    index = list(G.nodes).index(v)
    column = A[:, index] #column of vth node. which contains its in-links
    for edge in column:
        indeg += edge[0, 0]
    return indeg

def weighted_sum_outdegree(A, v, n, G): # (adjacency matrix, node)
    outdeg = 0
    index = list(G.nodes).index(v)
    row = A[index, :] #column of vth node. which contains its in-links
    for i in range(n):
        outdeg += row[0, i]
    return outdeg

def power_matrix(G, A):
    #(attempt to) calculate powers via augmented matrix
    n = G.number_of_nodes()
    A = A - np.identity(n) #A - I, adajacency matrix - identity matrix to get -1s in the diagonals.
    
    B = [0] * n #values vector
    for node in G.nodes():
        index = list(G.nodes).index(node)
        #print("index: " + str(index))
        B[index] =  (-1) * G.nodes[node]['value']

    try:
        X = np.linalg.solve(A, B)
    except: #if matrix has no solution, throw exception, to catch in main function
        print("NO SOLUTION")
        raise Exception("Augmented Powers Matrix is not linearly independant")

    #give the nodes the power property
    for node in G.nodes():
        index = list(G.nodes).index(node)
        G.add_node(node, power=X[index])
    return G


def influence_centrality(G, output_file_name = "output.csv"):
    #takes graph as input.
    #get adjacency matrix, it's used for a few things.
    Adj = nx.adjacency_matrix(G)
    A = Adj.todense()

    #CACLCULATE POWERS:
#   try: #attempt to calculate power using augemented power matrix
    G = power_matrix(G, A)
#    except: #system isn't linearly independant
#        print("power matrix function failed")
        #pass #it seems the matrix method solves all valid cases

    #the power functions added a "power" attribute to all nodes.
    #we just need to multiply by internal influence to get influence centrality
    max_centrality = 0 #keep track of most central node
    value_sum = 0
    for node in G.nodes:
        internal_influence = 1 - weighted_sum_indegree(A, node, G)
        infl_centrality = internal_influence * G.nodes[node]['power'] #influence centrality
        G.add_node(node, influence_centrality=infl_centrality, internal_influence=internal_influence)
        value_sum += G.nodes[node]['value']
        if infl_centrality > max_centrality:
            max_centrality = infl_centrality

    #compute centralization and standardize
    Centralization = 0
    centrality_sum = 0
    n = G.number_of_nodes()
    G_reaction_speed = 0
    G_stability = 0

    output = open(output_file_name, 'w', newline='')

    write = csv.writer(output)
    write.writerow(["node", "value", "influence", "standardized", "standardizedn", "reaction", "stability"])
    for node in G.nodes:
        inf_centrality = G.nodes[node]['influence_centrality']
        Centralization += (max_centrality - inf_centrality)
        centrality_sum += inf_centrality
        outdegree_sum = weighted_sum_outdegree(A, node, n, G)
        G.add_node(node, inf_centrality_standardized_1 = inf_centrality / value_sum,
                   inf_centrality_standardized_2 = inf_centrality / n)
        G.add_node(node, reaction_speed = G.nodes[node]['internal_influence'] * G.nodes[node]['inf_centrality_standardized_1'] ,
                   stability = (1 - (outdegree_sum / n)) * G.nodes[node]['inf_centrality_standardized_1'])
        G_stability += G.nodes[node]['stability']
        G_reaction_speed += G.nodes[node]['reaction_speed']
        write.writerow([node, round(G.nodes[node]['value'], 5), round(G.nodes[node]['influence_centrality'], 5),
            round(G.nodes[node]['inf_centrality_standardized_1'], 5), round(G.nodes[node]['inf_centrality_standardized_2'], 5),
            round(G.nodes[node]['reaction_speed'], 5), round(G.nodes[node]['stability'], 5)])
        
    Centralization_standardized_1 = Centralization / (value_sum * (n-1))
    Centralization_standardized_2 = Centralization / (n * (n-1))
    write.writerow(["G", round(value_sum,5), round(centrality_sum,5), round(Centralization_standardized_1, 5), round(Centralization_standardized_2,5),
        round(G_reaction_speed, 5), round(G_stability, 5)])
    
    output.close()
    return G
