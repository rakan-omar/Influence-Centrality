from pyvis.network import Network
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as pyplot
import itertools as it



def find_cycles(G, v, cycles, cylce_factors, skip_nodes = [], exclude_nodes=[], exclude_cycles=[]): #this is used for accounting for trails that include paths we have
    #skip_nodes consists of the nodes along the path we're examining, minus current node, since if they're along the path we're taking, we would have to repeat edges to use them (not a trail),
    #unless the nodes aren't adjacent in the cycle, then that's fine, but that would split into a path and a cycle from the first element (when calculating its power or power over it) and that's how we count them instead of as one cycle
    #exclude_node contaians ids of nodes we have already considered the cycles of for this path
    #last argument is indeces of cycles; so we don't count the current cycle infinitely many times, etc.
    
    #return the cycle_factors that 
    trail_factors = []
    cycles_to_exclude = [] #cycles we use in this iteration.
    for cycle_index in range(cycles_length(cycles)): #the cycle_factors have corresponding indices
        #if the cycle includes vertex v and hasn't been used and it doesn't inlcude a node to skip, add its cycle factor to trail factors
        cycle = cycles[cycle_index]
        if (v in cycle) and not (cycle_index in exclude_cycles):
            for node_to_skip in skip_nodes:
                if not node_to_skip in cycle:
                    exclude_cycles.append(cycle_index)
                    for node in cycle: #check for further cycles
                        if not (node in exclude_nodes): #but only if we haven't already check them for this node
                            exclude_nodes.append(node) #find_cycles calls itself. We don't want to repeat the calculations for the same cycles and nodes.
                            [new_trail_factors, cycles_used]  = find_cycles(G, v, cycles, cycle_factors, filter(lambda vertex: vertex != node, cycle) + skip_nodes, exclude_nodes, exclude_cycles)
                            new_trail_factors = get_combinations(new_trail_factors)
                    trail_factors.append((1 + sum(new_trail_factors)) * cycle_factors[cycle_index]) #current cycle factor, along with all cycles it could contain
    #add the products of all possible combinations of cycle factors from the ones used to trail factors. (a trail could contain any combination of cycles found along the path)    
    return [trail_factors, exclude_cycles]



#this function finds the sum of the product of all trails from on 
def sum_products_u_v_trails(G, u, v, A, cycles, cycle_factors): #(Graph, source node, destination node, Adjacency matrix, all cycles (networkx has a function for this), cycle factors)
    #NOTE, this function excludes all trails containing our source node.
    Sum = 0
    for path in nx.all_simple_paths(G, u, v): #iterate over paths, then we check for trails that contain them along the way
        path_factor = 1 #the product of the weights of the edges along this path
        non_path_trails = [] #here we append the factors of the trails that contain this path (without the path factor as a component, for now)
        cycles_to_exclude = [] #we don't count the same cycle multiple times along the same path.
        nodes_to_exclude = [u] #nodes we already considered along this path, so we don't consider them again when looking for cycles.
        for node_index in range(len(path)): #for each node, check trails that diverge from this once
            if (node_index < len(path) - 1): #we stop getting edges at the last node, this is why we use the index
                edge = J.get_edge_data(path[node_index], path[index + 1])
                path_factor = path_factor * edge['weight']
            #we don't want to get cycles for first node (or those that include first node) now, those are in the denominator
            #we also don't want cycles that contain other nodes in the same path, since if they're along the path we're taking, we would have to repeat edges to use them (not a trail)
            nodes_to_exclude.append[path[node_index]]
            [new_trail_factors, cycles_used] = find_cycles(G, path[node_index], cycles, cycle_factors, path[:node_index] + path[node_index+1:], nodes_to_exclude, cycles_to_exclude)
            non_path_trails.extend(new_trail_factors)
            cycles_to_exclude.extend(cycles_used)
        #take combinations also
        non_path_trails = get_combinations(non_path_trails)  
        Sum += (1 + sum(non_path_trails)) * path_factor #add factors of these trails to the sum of the factors all trails from u to v, which is what we're caclulating
    return Sum




def sum_product_of_circuits(G, u, A, cycles, cycle_factors): #this is for trail circuits from node u back to itself
    trail_factors = []
    cycles_used = []
    nodes_used = [u]
    #cycles containing u
    for cycle_index in range(len(cycles)):
        #for each node in the cycle
        if u in cycles[cycle_index]:                
            cycles_used.append(cycle_index)
            new_factors = []
            for node in cycles[cycle_index]:
                cycle = cycles[cycle_index][:] #copy not reference
                cycle.remove(node)
                nodes_used.append(node)
                [cycle_trail_factors, cycles_in_cycles] = find_cycles(G, v, cycles, cycle_factors, cycle, nodes_used, cycles_used)
                new_factors.append(cycle_trail_factors)
                cycles_used.extend(cycles_in_cycles)
                cycles_used.extend(cycles_in_cycles)
            new_factors = get_combinations(new_factors)
            trail_factors.append((1 + sum(new_factors)) * cycle_factors[cycle_index])
        Sum = sum(trail_factors)
    return Sum

#product of the weights of the edges of each cyle 
def calculate_cycle_factors(G, cycles):
    cycle_factors = [1] * cycles_length(cycles)

    for cycle_index in range(cycles_length(cycles)):
        cycle = cycles[cycle_index]
        for edge_index in range(len(cycle)): #traverse the edges of each cycle
            if (edge_index < len(cycle) - 1):#the last node we have to use a different index
                next_index = edge_index + 1
            else:
                next_index = 0

            edge = G.get_edge_data(cycle[edge_index], cycle[next_index])
            cycle_factors[cycle_index] = edge['weight'] * cycle_factors[cycle_index]
            
    return cycle_factors


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
    cycle_factors = calculate_cycle_factors(G, cycles)
        
    #CACLCULATE POWERS:
    try: #attempt to calculate power using augemented power matrix
        G = power_matrix(G, A)
    except: #system isn't linearly independant. use all trails formula
        G = power_trails(G, A)

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
                   inf_centrality_standardized_2 = inf_centrality / n, reaction_speed = G.nodes[node]['internal_influence'] * inf_centrality,
                   stability = (1 - (outdegree_sum / n)) * inf_centrality)
        G_stability += G.nodes[node]['stability']
        G_reaction_speed += G.nodes[node]['reaction_speed']
        #print("node " + str(node) + ":  " + str(G.nodes[node]) + "\n")
        
        print("%5d & %10.2f & %10.9f & %10.9f & %10.9f & %10.9f & %10.9f" % (node, G.nodes[node]['value'], G.nodes[node]['influence_centrality'],
                                                        G.nodes[node]['inf_centrality_standardized_1'], G.nodes[node]['inf_centrality_standardized_2'],
                                                        G.nodes[node]['reaction_speed'], G.nodes[node]['stability']))
    Centralization_standardized_1 = Centralization / (value_sum * (n-1))
    Centralization_standardized_2 = Centralization / (n * (n-1))
    G_reaction_speed = G_reaction_speed / centrality_sum
    G_stability = G_stability / centrality_sum
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
