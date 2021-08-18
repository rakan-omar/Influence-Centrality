from pyvis.network import Network
import networkx as nx
import numpy as np
import scipy as sp
import random as rd
import matplotlib.pyplot as pyplot
import csv
import math
import influence_centrality as ic


#North_America = ["ATG", "ABW", "BHS", "BRB", "BLZ", "CAN", "CRI", "CUB", "DMA", "DOM",
#                 "SLV", "GRD","GTM", "HTI", "HND", "JAM", "MEX", "NIC", "PAN",
#                 "KNA", "LCA", "VCT", "TTO", "USA", "BMU", "CYM", "CUW",
#                 "GRL", "SXM", "TCA"] #Virgin islands and puerto rico aren't listed separately for imports exports.

Selected_Countries = ["USA", "CAN", "FRA", "GBR", "MEX", "DEU", "ARG"]

def draw_G(G, filename = 'network_graph'):
    #loop over all nodes, give size and labels
    for node in G.nodes:
        G.add_node(node, size=math.ceil(G.nodes[node]['value'] * 15), physics=False)
    graph = Network('700px', '700px', directed=True)
    graph.from_nx(G)
    graph.set_edge_smooth("dynamic")
    for edge in graph.edges:
        edge['label'] = edge['weight']
    graph.show(filename + '.html')
    return graph


G = nx.DiGraph()
#create network based on CSV file contents;
input_f = open('Trade_BEH0_2018_Import_2020Jul10.csv', 'r')
reader = csv.reader(input_f)
next(reader) #header
for row in reader:
    #ADJUST COLUMNS BASED ON INPUT FILE
    if (row[2] != "WLD" and row[9] != "WLD"): #if data for 'world' is in the file, don't include it.
    #if importing or exporting country is in Selected_Countries, it has a node, otherwise, it's included in Rest of the World (RotW).
        arrow_from = 'RotW'
        arrow_to = 'RotW'
        if (row[2] in Selected_Countries):
            arrow_from = row[2]

        if (row[9] in Selected_Countries):
            arrow_to = row[9]
        
        if (arrow_from != arrow_to): #found two loops, plus the one on world, throwing off the calculcations
            total_trade_value = float(row[5])
            #if row[6] != '':
            #    total_trade_value *= float(row[6])
            #elif row[8] != '':
            #    total_trade_value *= float(row[8])


            #if edge already exists, add total_trade_value to (temporary) weight.
            if G.has_edge(arrow_from, arrow_to):
                #previous weight
                old_weight = G.get_edge_data(arrow_from, arrow_to)['weight']
                #new weight
                G.add_edge(arrow_from, arrow_to, weight=(total_trade_value + old_weight))
            else:
                G.add_edge(arrow_from, arrow_to, weight=total_trade_value)


input_f.close()
values = open('GDP_2018.csv')
max_gdp = 0
#min_gdp = 999999999999.9
read = csv.reader(values)
next(read)
n = G.number_of_nodes()
for row in read:
    if (row[1] in Selected_Countries):
        node_index = n
        try:
            node_index = list(G.nodes).index(row[1])
        except:
            print("NOT FOUND:" + row[0] + " , " + row[1])
            continue
        GDP = float(row[2])
        G.nodes[row[1]]['value'] = GDP
        if GDP > max_gdp:
            max_gdp = GDP
        #if GDP < min_gdp:
        #    min_gdp = GDP
    elif (row[1] != "WLD"): #not in Selected_Countries and not world, include in rotw
        old_value = 0
        try:
            old_value = G.nodes["RotW"]['value']
        except:
            pass
        G.nodes["RotW"]['value'] = old_value + float(row[2])

if G.nodes["RotW"]["value"] > max_gdp:
    max_gdp = G.nodes["RotW"]["value"]

input_f.close()



#standardize edge weights, by dividing by value of destination node
for edge in G.edges:
    #check if there is an opposite edge, curve it
    edge_weight = G.get_edge_data(edge[0], edge[1])['weight'] *1000 #imports exports listed in millions of dollars. Gdp in dollars.
    if "value" in G.nodes[edge[1]]:
        G.add_weighted_edges_from([(edge[0], edge[1], edge_weight/G.nodes[edge[1]]['value'])])
    else:
        G.add_weighted_edges_from([(edge[0], edge[1], edge_weight/max_gdp)])

for node in G.nodes:
    if "value" in G.nodes[node]:
        G.nodes[node]['value'] = G.nodes[node]['value']/max_gdp
    else: #this line is included for incomplete data
        G.nodes[node]['value'] = min_gdp/max_gdp

print("node values standardized over: " + str(max_gdp))
#for node in G.nodes:
#    print(node + ": " + str(G.nodes[node]['value']))
#for edge in G.edges:
#    print(edge[0] + " to " + edge[1] + ": " +str(G.get_edge_data(*edge)['weight']))
ic.influence_centrality(G, 'imports.csv')
draw_G(G, 'imports')
