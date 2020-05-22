import networkx as nx
import numpy as np
import random as rn
import kdegree as kd
import time

G = nx.read_edgelist('test graphs/PT.txt',nodetype=int)
#G = nx.relabel_nodes(G, lambda x: x-1)
#G = nx.barabasi_albert_graph(1000,3)
#G = nx.erdos_renyi_graph(1000,0.01)
#G = nx.watts_strogatz_graph(1000,10,0.1)
#G = nx.read_gpickle('test graphs/ws_test_1.gpkl')

print(nx.number_of_nodes(G),nx.number_of_edges(G),nx.density(G))

noise=10
k=30
start = time.time()
Ga = kd.graph_anonymiser(G,k=k,algorithm=1,noise=noise,greedy_anonymiser=False)
print("Total execution time =",time.time()-start)

H = nx.intersection(G,Ga)

num_edges_in_G = len(set(G.edges()))
num_edges_in_both = len(set(H.edges()))

print("Edges overlap = " + str(100*num_edges_in_both/num_edges_in_G)+"%")
print("Num edges original graph = " + str(nx.number_of_edges(G)))
print("Num edges anonymised graph = " + str(nx.number_of_edges(Ga)))
