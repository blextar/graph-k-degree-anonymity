# Based on https://dl.acm.org/doi/10.1145/1376616.1376629

import networkx as nx
import numpy as np
import random as rn

def get_degree_sequence(G):
    return [d for n, d in G.degree()]
    
def assignment_cost(degree_sequence):
    return np.sum(np.array(degree_sequence[0])-np.array(degree_sequence))
    
# NOTE: not implemented yet!!
def degree_anonymiser(degree_sequence,k):
    l = len(degree_sequence)
    if l < 2*k:
        anonymisation_cost = assignment_cost(degree_sequence)
    else:
        anonymisation_cost = np.min([0,assignment_cost(degree_sequence)])
    return anonymised_degree_sequence
    
def greedy_degree_anonymiser(degree_sequence,k):
    # create a k-anonymomus group with all nodes having degree equal to the largest in the group
    # degree_sequence is sorted, so the largest element is the first one
    largest_degree = degree_sequence[0]

    # if there are less than 2*k nodes left to anonymise, group them together and finish
    if len(degree_sequence) < 2*k:
        anonymised_sequence = [largest_degree]*len(degree_sequence)
        return anonymised_sequence
    
    # otherwise start a new group
    anonymised_sequence = [largest_degree]*k
    
    # then check if we can add another node to the current group or if it's better to create a new group
    c_merge = degree_sequence[0]-degree_sequence[k]+assignment_cost(degree_sequence[k+1:2*k])
    c_new = assignment_cost(degree_sequence[k:2*k-1])
    
    # keep adding nodes while you can
    i = 0
    while c_new >= c_merge:
        # add current node
        anonymised_sequence = anonymised_sequence + [largest_degree]
        # checking the next node now..
        i = i+1
        # if there is only one node left, add it to the group
        if k+i == len(degree_sequence)-1:
            anonymised_sequence = anonymised_sequence + [largest_degree]    
            return anonymised_sequence
        if 2*k+i >= len(degree_sequence):
            c_merge = np.sum(np.array(largest_degree)-np.array(degree_sequence[k+i:]))
            c_new = assignment_cost(degree_sequence[k+i:])
            if c_new >= c_merge:
                anonymised_sequence = anonymised_sequence + [largest_degree]*len(degree_sequence[k+i:])
                return anonymised_sequence
        else:
            c_merge = largest_degree-degree_sequence[k+i]+assignment_cost(degree_sequence[k+i+1:2*k+i])
            c_new = assignment_cost(degree_sequence[k+i:2*k+i-1])
        
    # when you stop adding new nodes, make a recursive call starting a new group on the remaining sequence
    anonymised_sequence = anonymised_sequence + greedy_degree_anonymiser(degree_sequence[k+i:],k)
    return anonymised_sequence

def construct_graph(degree_sequence):
    n = len(degree_sequence)
    # if sum of the degree sequence is odd, the degree sequence isn't realisable
    if np.sum(degree_sequence) % 2 != 0:
        return None
    
    G = nx.empty_graph(n)
    
    # transform list of degrees in list of (vertex, degree)
    vd = [(v,d) for v,d in enumerate(degree_sequence)]
    
    while True:
        # sort the list of pairs by degree (second element in the pair)
        vd.sort(key=lambda tup: tup[1], reverse=True)
        # if we ended up with a negative degree, the degree sequence isn't realisable
        if vd[-1][1] < 0:
            return None
        
        tot_degree = 0
        for vertex in vd:
            tot_degree = tot_degree + vertex[1]
        # if all the edges required by the degree sequence have been added, G has been created
        if tot_degree == 0:
            return G
        
        # gather all the vertices that need more edges 
        remaining_vertices = [i for i,vertex in enumerate(vd) if vertex[1] > 0]
        # shuffle them
        idx = remaining_vertices[rn.randrange(len(remaining_vertices))]
        # pick a random one
        v = vd[idx][0]
        
        # iterate over all the vertices (vd is sorted from largest to smallest)
        for i,u in enumerate(vd):
            # stop when we added v_d edges
            if vd[idx][1] == 0:
                break
            # don't add self-loops
            if u[0] == v:
                continue
            # add an edge
            G.add_edge(v,u[0])
            # decrease the degree of the connected vertex
            vd[i] = (u[0],u[1] - 1)
            # keep track of how many edges we added
            vd[idx] = (vd[idx][0],vd[idx][1] - 1)
            
def find_max_swap(G,edges_orig):
    edges = G.edges
    num_samples = int(np.floor(np.log(len(edges))))
    selected_edges = rn.sample(edges, k=num_samples) #change to k=num_samples
    best_swap = (-1e8,None)
    for i in range(len(selected_edges)-1):
        for j in range(i+1,len(selected_edges)):
            e1 = selected_edges[i]
            e2 = selected_edges[j]
            if (e1[0],e2[0]) not in edges and (e1[1],e2[1]) not in edges:
                c = 0
                c = c - 1 if e1 in edges_orig else c
                c = c - 1 if e2 in edges_orig else c
                c = c + 1 if (e1[0],e2[0]) in edges_orig else c
                c = c + 1 if (e1[1],e2[1]) in edges_orig else c
                if c > best_swap[0]:
                    best_swap = (c,(e1,e2,(e1[0],e2[0]),(e1[1],e2[1])))
            if (e1[0],e2[1]) not in edges and (e1[1],e2[0]) not in edges:
                c = 0
                c = c - 1 if e1 in edges_orig else c
                c = c - 1 if e2 in edges_orig else c
                c = c + 1 if (e1[0],e2[1]) in edges_orig else c
                c = c + 1 if (e1[1],e2[0]) in edges_orig else c
                if c > best_swap[0]:
                    best_swap = (c,(e1,e2,(e1[0],e2[1]),(e1[1],e2[0])))
    return best_swap
    
def greedy_swap(G, G_orig):
    edges_orig = G_orig.edges()
    G_new = G.copy()
    (c, (e1, e2, ee1, ee2)) = find_max_swap(G_new,edges_orig)
    while c > 0:
        G_new.remove_edge(e1[0],e1[1])
        G_new.remove_edge(e2[0],e2[1])
        G_new.add_edge(ee1[0],ee1[1])
        G_new.add_edge(ee2[0],ee2[1])
        (c, (e1, e2, ee1, ee2)) = find_max_swap(G_new,edges_orig)
    return G_new
    
def priority(degree_sequence,G_orig):
    
    target_edges = G_orig.edges()

    n = len(degree_sequence)
    # if sum of the degree sequence is odd, the degree sequence isn't realisable
    if np.sum(degree_sequence) % 2 != 0:
        return None
    
    G = nx.empty_graph(n)
    
    # transform list of degrees in list of (vertex, degree)
    vd = [(v,d) for v,d in enumerate(degree_sequence)]
    
    while True:
        # sort the list of pairs by degree (second element in the pair)
        vd.sort(key=lambda tup: tup[1], reverse=True)
        # if we ended up with a negative degree, the degree sequence isn't realisable
        if vd[-1][1] < 0:
            return None
        
        tot_degree = 0
        for vertex in vd:
            tot_degree = tot_degree + vertex[1]
        # if all the edges required by the degree sequence have been added, G has been created
        if tot_degree == 0:
            return G
        
        # gather all the vertices that need more edges 
        remaining_vertices = [i for i,vertex in enumerate(vd) if vertex[1] > 0]
        # pick a random one
        idx = remaining_vertices[rn.randrange(len(remaining_vertices))]
        # pick a random one
        v = vd[idx][0]
                
        second_choices = list()
        # iterate over all the vertices u such that (u,v) is an edge in the original graph
        for i,u in enumerate(vd):
            # stop when we added v_d edges
            if vd[idx][1] == 0:
                break
            # don't add self-loops
            if u[0] == v:
                continue
            # add an edge
            if (v,u[0]) in target_edges and u[1] > 0:
                G.add_edge(v,u[0])
                # decrease the degree of the connected vertex
                vd[i] = (u[0],u[1] - 1)
                # keep track of how many edges we added
                vd[idx] = (vd[idx][0],vd[idx][1] - 1)
            else:
                second_choices.append(i)
        # now consider also other vertices in vd, if we still need to add edges
        for u_idx in second_choices:
            # stop when we added v_d edges
            if vd[idx][1] == 0:
                break
            # don't add self-loops
            if vd[u_idx][0] == v:
                continue
            # add an edge
            G.add_edge(v,vd[u_idx][0])
            # decrease the degree of the connected vertex
            vd[u_idx] = (vd[u_idx][0],vd[u_idx][1] - 1)
            # keep track of how many edges we added
            vd[idx] = (vd[idx][0],vd[idx][1] - 1)
            
def graph_anonymiser(G,k,with_priority=True):
    degree_sequence = get_degree_sequence(G)
    degree_sequence.sort(reverse=True)
    anonymised_sequence = greedy_degree_anonymiser(degree_sequence,k)
    if with_priority:
        Ga = priority(anonymised_sequence,G)
    else:
        Ga = construct_graph(anonymised_sequence)
    while Ga is None:
        candidates = list(nx.non_edges(G))
        e = rn.sample(candidates, k=1)
        G.add_edge(e[0][0],e[0][1])
        degree_sequence = get_degree_sequence(G)
        degree_sequence.sort(reverse=True)
        anonymised_sequence = greedy_degree_anonymiser(degree_sequence,k)
        if with_priority:
            Ga = priority(anonymised_sequence,G)
        else:
            Ga = construct_graph(anonymised_sequence)
    return Ga