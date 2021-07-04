from pyvis.network import Network
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as pyplot
import itertools as it


def weighted_sum_indegree(A, v): # (adjacency matrix, node)
    indeg = 0
    column = A[:, v - 1] #column of vth node. which contains its in-links
    for edge in column:
        indeg += edge[0, 0]
    return indeg

def weighted_sum_outdegree(A, v, n): # (adjacency matrix, node)
    outdeg = 0
    row = A[v-1, :] #column of vth node. which contains its in-links
    for i in range(n):
        outdeg += row[0, i]
    return outdeg

def power_matrix(G, A):
    #(attempt to) calculate powers via augmented matrix
    n = G.number_of_nodes()
    A = A - np.identity(n) #A - I, adajacency matrix - identity matrix to get -1s in the diagonals.

    B = [0] * n #values vector

    for node in G.nodes():
        B[node - 1] =  (-1) * G.nodes[node]['value']

    try:
        X = np.linalg.solve(A, B)
    except: #if matrix has no solution, throw exception, to catch in main function
        raise Exception("Augmented Powers Matrix is not linearly independant")

    #give the nodes the power property
    for node in G.nodes():
        G.add_node(node, power=X[node - 1])

    print("augmented matrix was used to calculate powers")
    return G


def power_trails(G, A, cycles):
    #calculate powers using all paths method.

    for power_node in G.nodes: #the node we're calculating the power of
        power = G.nodes[power_node]['value']

        #the sum in the numerator: for each node, for all trails to it ....
        for destination_node in G.nodes:
            if (destination_node != power_node): #it's important to exclude the power node, as well as trails that go through 
                sum_product_of_trails = sum_products_u_v_trails(G, power_node, destination_node, A, cycles, cycle_factors)
                power += sum_product_of_trails * G[destination_node]['value']
        power_denominator = 1 - sum_product_of_circuits(G, power_node, A, cycles, cycle_factors)
        if (power_denominator != 0):
            power = power / power_denominator
        else: #special case
            pass
            #denominator is 0 means it's a componenet of cycle(s), consisting (mostly) of weight 1 edges.
            #if the denominator is zero, they'll get influence centrality 0 or undefined.
            

        G.add_node(power_node, power=power) #assign the power as a node attribute

        print("all trails method was used to caclculate powers")
    return G




def influence_centrality(G):
    #takes graph as input.
    #get adjacency matrix, it's used for a few things.
    Adj = nx.adjacency_matrix(G)
    A = Adj.todense()

    #cycle factors are the products of the weights of the edges in each cycle.
    cycles = nx.simple_cycles(G)
    #cycle_factors = calculate_cycle_factors(G, cycles)
        
    #CACLCULATE POWERS:
    try: #attempt to calculate power using augemented power matrix
        G = power_matrix(G, A)
    except: #system isn't linearly independant. use all trails formula
        pass #it seems the matrix method solves all valid cases so nvm

    #the power functions added a "power" attribute to all nodes.
    #we just need to multiply by internal influence to get influence centrality
    max_centrality = 0 #keep track of most central node
    value_sum = 0
    for node in G.nodes:
        internal_influence = 1 - weighted_sum_indegree(A, node)
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

    print("%5s & %10s & %10s & %10s & %10s & %10s & %10s" % ("node (i)", "V(v_i)", "C_I(v_i)", "C_I'(v_i) (standardized over values)",
                                    "C_I''(v_i) (standardized over n)", "Reaction speed", "Stability"))
    
    for node in G.nodes:
        inf_centrality = G.nodes[node]['influence_centrality']
        Centralization += (max_centrality - inf_centrality)
        centrality_sum += inf_centrality
        outdegree_sum = weighted_sum_outdegree(A, node, n)
        G.add_node(node, inf_centrality_standardized_1 = inf_centrality / value_sum,
                   inf_centrality_standardized_2 = inf_centrality / n)
        G.add_node(node, reaction_speed = G.nodes[node]['internal_influence'] * G.nodes[node]['inf_centrality_standardized_1'] ,
                   stability = (1 - (outdegree_sum / n)) * G.nodes[node]['inf_centrality_standardized_1'])
        G_stability += G.nodes[node]['stability']
        G_reaction_speed += G.nodes[node]['reaction_speed']
        #print("node " + str(node) + ":  " + str(G.nodes[node]) + "\n")
        
        print("%5d & %10.2f & %10.9f & %10.9f & %10.9f & %10.9f & %10.9f" % (node, G.nodes[node]['value'], G.nodes[node]['influence_centrality'],
                                                        G.nodes[node]['inf_centrality_standardized_1'], G.nodes[node]['inf_centrality_standardized_2'],
                                                        G.nodes[node]['reaction_speed'], G.nodes[node]['stability']))
    Centralization_standardized_1 = Centralization / (value_sum * (n-1))
    Centralization_standardized_2 = Centralization / (n * (n-1))
    G_reaction_speed = G_reaction_speed / (centrality_sum * value_sum)
    G_stability = G_stability / (centrality_sum * value_sum)
    print("%5s & %10.2f & %10s & %10.9f & %10.9f & %10.9f & %10.9f" % ("G", value_sum, "-", Centralization_standardized_1, Centralization_standardized_2,
                                                                                G_reaction_speed, G_stability))
    #print("Standardized centralization 1: %10.5f \n Standardized centralization 2: %10.5f \n reaction speed: %10.5f \n stability: %10.5f" % (Centralization_standardized_1,
           # Centralization_standardized_2, G_reaction_speed, G_stability))
    print("SUM OF VALUES: %10.5f" % value_sum)
    print("SUM OF CENTRALITIES: %10.5f" % centrality_sum)
    return G





#"cycles" is given as a datatype called "generator", not a list/array, there isn't a built in method for getting its length
def cycles_length(cycles):
    length = 0
    for cycle in cycles:
        length += 1
    return length



#get trails produced by combining paths and cycles in different ways, when possible
def get_combinations(trails):
    new_trail_combinations = []
    for n in range(len(trails)):
        for combination in it.combinations(trails, n + 1):
            #take the products of the combinations.
            new_trail_combinations.append(np.prod(combination))
    trails = trails + new_trail_combinations
    return trails
