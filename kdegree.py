# Based on Liu & Terzi's k-degree anonymity:
# [1] https://dl.acm.org/doi/10.1145/1376616.1376629

import networkx as nx
import numpy as np
import random as rn

def sort_dv(dv):
    dv.sort(reverse=True)
    # permutation holds the mapping between original vertex and degree-sorted vertices
    degree_sequence,permutation = zip(*dv)
    degree_sequence = np.array(degree_sequence)
    return degree_sequence,permutation

# "degree anonymization cost" as defined in Section 4 of [1] 
def assignment_cost_additions_only(degree_sequence):
    return np.sum(degree_sequence[0]-degree_sequence)
    
# median of a sorted array
def median(degree_sequence):
    n = len(degree_sequence)
    return degree_sequence[n//2] if n % 2 else (degree_sequence[n//2-1]+degree_sequence[n//2])//2
    
# "degree anonymization cost" as defined in Section 8 of [1]
def assignment_cost_additions_deletions(degree_sequence):
    md = median(degree_sequence)
    return np.sum(np.abs(md-degree_sequence))
    
# Precomputation of the anonymisation cost, as described in Section 4 of [1]
def anonymisation_cost_precomputation(degree_sequence,k,with_deletions):
    n = np.size(degree_sequence)
    C = np.full([n,n],np.inf)
    for i in range(n-1):
        for j in range(i+k-1,np.min([i+2*k,n])):
            if with_deletions:
                if C[i,j-1] == np.inf:
                    C[i,j] = assignment_cost_additions_deletions(degree_sequence[i:j+1])
                else:
                    C[i,j] = C[i,j-1] +  median(degree_sequence[i:j+1]) - degree_sequence[j]
            else:
                if C[i,j-1] == np.inf:
                    C[i,j] = assignment_cost_additions_only(degree_sequence[i:j+1])
                else:
                    C[i,j] = C[i,j-1] + degree_sequence[i] - degree_sequence[j]
    return C
    
# The dynamic programming algorithm described in Section 4 of [1]
def dp_degree_anonymiser(degree_sequence,k,with_deletions=False):
    C = anonymisation_cost_precomputation(degree_sequence,k,with_deletions)
    n = np.size(degree_sequence)
    Da = np.full(n,np.inf)
    sequences = [None] * n
    cost, anonymised_sequence = dp_degree_anonymiser_recursion(degree_sequence,k,C,n,Da,sequences,with_deletions)
    return anonymised_sequence

# The dynamic programming algorithm described in Section 4 of [1] - recursion part
def dp_degree_anonymiser_recursion(degree_sequence,k,C,n,Da,sequences,with_deletions):
    
    group_degree = None
    if with_deletions:
        group_degree = median(degree_sequence)
    else:
        group_degree = degree_sequence[0]
    all_in_one_group_sequence = np.full(n,group_degree)
    all_in_one_group_cost = C[0,n-1]  
      
    if n < 2*k:
        return all_in_one_group_cost, all_in_one_group_sequence
    else:
        
        min_cost = np.inf
        min_cost_sequence = np.empty(0)
        # number of recursions optimised according to Eq. 4 in [1]
        # originally: range(k-1,n-k)
        for t in range(np.max([k-1,n-2*k]),n-k):
            # this IF-ELSE is to avoid recomputing cost and sequence for the same value of t more than once
            if Da[t] == np.inf:
                cost, sequence = dp_degree_anonymiser_recursion(degree_sequence[0:t+1],k,C,t+1,Da,sequences,with_deletions)
                Da[t] = cost
                sequences[t] = sequence
            else:
                cost = Da[t]
                sequence = sequences[t]
            cost = cost + C[t+1,n-1]
            if cost < min_cost:
                min_cost = cost
                if with_deletions:
                    min_cost_sequence = np.concatenate((sequence,np.full(np.size(degree_sequence[t+1:]),median(degree_sequence[t+1:]))))
                else:
                    min_cost_sequence = np.concatenate((sequence,np.full(np.size(degree_sequence[t+1:]),degree_sequence[t+1])))                
        to_return = (min_cost, min_cost_sequence) if min_cost < all_in_one_group_cost else (all_in_one_group_cost, all_in_one_group_sequence)
        return to_return

# Section 6.2 in [1]
def priority(degree_sequence,original_G):

    n = len(degree_sequence)
    # if the sum of the degree sequence is odd, the degree sequence isn't realisable
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
        v = vd[idx][0]
        # iterate over all the degree-sorted vertices u such that (u,v) is an edge in the original graph
        for i,u in enumerate(vd):
            # stop when we added v_d edges
            if vd[idx][1] == 0:
                break
            # don't add self-loops
            if u[0] == v:
                continue
            # make sure we're not adding the same edge twice..
            if G.has_edge(u[0],v):
                continue
            # add the edge if this exists also in the original graph
            if original_G.has_edge(v,u[0]) and u[1] > 0:
                G.add_edge(v,u[0])
                # decrease the degree of the connected vertex
                vd[i] = (u[0],u[1] - 1)
                # keep track of how many edges we added
                vd[idx] = (v,vd[idx][1] - 1)
                
        # iterate over all the degree-sorted vertices u such that (u,v) is NOT an edge in the original graph
        for i,u in enumerate(vd):
            # stop when we added v_d edges
            if vd[idx][1] == 0:
                break
            # don't add self-loops
            if u[0] == v:
                continue
            # make sure we're not adding the same edge twice..
            if G.has_edge(v,u[0]):
                continue
            # now add edges that are not in the original graph
            if not original_G.has_edge(v,u[0]):
                G.add_edge(v,u[0])
                # decrease the degree of the connected vertex
                vd[i] = (u[0],u[1] - 1)
                # keep track of how many edges we added
                vd[idx] = (v,vd[idx][1] - 1)

def probing(dv,noise):    
    # increase only the degree of the lowest degree nodes, as suggested in the paper
    n = len(dv)
    for v in range(-noise,0):
        dv[v] = (np.min([dv[v][0]+1,n-1]),dv[v][1])
    return dv

# Anonymise G given a value of k using DP degree anonymiser, PRIORITY, and PROBING (edge removals: optional)
def graph_anonymiser(G,k,noise=1,with_deletions=False):
    
    dv = [(d,v) for v, d in G.degree()]
    degree_sequence,permutation = sort_dv(dv)
        
    attempt = 1
    print("Attempt number",attempt)
    anonymised_sequence = dp_degree_anonymiser(degree_sequence,k,with_deletions=with_deletions)
    
    new_anonymised_sequence = [None] * len(degree_sequence)
    for i in range(len(permutation)):
        new_anonymised_sequence[permutation[i]] = anonymised_sequence[i]
    anonymised_sequence = new_anonymised_sequence
    
    Ga = priority(anonymised_sequence,G)
    
    while Ga is None:
        attempt = attempt+1
        print("Attempt number",attempt)
        
        dv = probing(dv,noise)
        degree_sequence,permutation = sort_dv(dv)
        
        anonymised_sequence = dp_degree_anonymiser(degree_sequence,k,with_deletions=with_deletions)
        
        new_anonymised_sequence = [None] * len(degree_sequence)
        for i in range(len(permutation)):
            new_anonymised_sequence[permutation[i]] = anonymised_sequence[i]
        anonymised_sequence = new_anonymised_sequence
        
        if not nx.is_valid_degree_sequence_erdos_gallai(anonymised_sequence):
            continue
        Ga = priority(anonymised_sequence,G)
        if Ga is None:
            print("the sequence is valid but the graph construction failed")
            
    return Ga