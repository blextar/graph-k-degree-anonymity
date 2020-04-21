# Based on Liu & Terzi's k-degree anonymity:
# [1] https://dl.acm.org/doi/10.1145/1376616.1376629

import networkx as nx
import numpy as np
import random as rn

def get_degree_sequence(G):
    return [d for n, d in G.degree()]

# "degree anonymization cost" as defined in Section 4 of [1] 
def assignment_cost_additions_only(degree_sequence):
    return np.sum(np.array(degree_sequence[0])-np.array(degree_sequence))
    
# "degree anonymization cost" as defined in Section 8 of [1]
def assignment_cost_additions_deletions(degree_sequence):
    return np.sum(np.abs(np.int(np.median(degree_sequence))-np.array(degree_sequence)))
    
# Precompuation of the anonymisation cost, as described in Section 4 of [1]
# Can be further optimise to consider a smaller range of j
def anonymisation_cost_precompuation(degree_sequence,k,deletions):
    n = len(degree_sequence)
    C = np.full([n,n],np.inf)
    for i in range(n-1):
        for j in range(i+1,n):
            if deletions:
                C[i,j] = assignment_cost_additions_deletions(degree_sequence[i:j+1])
            else:
                C[i,j] = assignment_cost_additions_only(degree_sequence[i:j+1])
    return C
    
# The dynamic programming algorithm described in Section 4 of [1]
# Significantly slower than the greedy approach
def dp_degree_anonymiser(degree_sequence,k,deletions=False):
    C = anonymisation_cost_precompuation(degree_sequence,k,deletions)
    cost, anonymised_sequence = dp_degree_anonymiser_recursion(degree_sequence,len(degree_sequence),k,C,deletions)
    return anonymised_sequence

# The dynamic programming algorithm described in Section 4 of [1] - recursion part
def dp_degree_anonymiser_recursion(degree_sequence,end_idx,k,C,deletions):
    n = len(degree_sequence)
    group_degree = degree_sequence[0]
    if deletions:
        all_in_one_group_sequence = [np.int(np.median(degree_sequence[0:end_idx]))]*n
    else:
        all_in_one_group_sequence = [group_degree]*n
    #all_in_one_group_cost = assignment_cost_additions_only(degree_sequence)
    all_in_one_group_cost = C[0,end_idx-1]    
    if n < 2*k:
        return all_in_one_group_cost, all_in_one_group_sequence
    else:
        costs = []
        sequences = []
        # number of recursions optimised according to Eq. 4 in [1]
        # originally: range(k-1,n-k)
        for t in range(np.max([k-1,n-2*k]),n-k):
            cost, sequence = dp_degree_anonymiser_recursion(degree_sequence[0:t+1],t+1,k,C,deletions)
            #cost = cost + assignment_cost_additions_only(degree_sequence[t+1:])
            cost = cost + C[t+1,end_idx-1]
            costs.append(cost)
            if deletions:
                sequences.append(sequence + [np.int(np.median(degree_sequence[t+1:end_idx-1]))]*len(degree_sequence[t+1:]))
            else:
                sequences.append(sequence + [degree_sequence[t+1]]*len(degree_sequence[t+1:]))
        min_indices = np.where(costs == np.amin(costs))
        min_idx = min_indices[0][0]
        min_cost = costs[min_idx]
        min_sequence = sequences[min_idx]
        to_return = (min_cost, min_sequence) if min_cost < all_in_one_group_cost else (all_in_one_group_cost, all_in_one_group_sequence)
        return to_return
       
# The dynamic programming algorithm described in Section 8 of [1] 
def dp_degree_anonymiser_with_deletions(degree_sequence,k):
    l = len(degree_sequence)
    if l < 2*k:
        anonymisation_cost = assignment_cost_additions_only(degree_sequence)
    else:
        anonymisation_cost = np.min([0,assignment_cost_additions_only(degree_sequence)])
    return anonymised_degree_sequence

# The greedy algorithm described in Section 4 of [1] 
def greedy_degree_anonymiser(degree_sequence,k):
    # create a k-anonymomus group with all nodes having degree equal to the largest in the group
    # degree_sequence is sorted, so the largest element is the first one
    group_degree = degree_sequence[0]
    n = len(degree_sequence)
        
    # if there are less than 2*k nodes left to anonymise, group them together and finish
    if n < 2*k:
        anonymised_sequence = [group_degree]*n
        return anonymised_sequence
    # otherwise start a new group
    anonymised_sequence = [group_degree]*k
    # then check if we can add another node to the current group or if it's better to create a new group
    # cost of grouping the next k nodes together (we know we have at least k nodes left, thanks to the IF above)
    c_new = assignment_cost_additions_only(degree_sequence[k:2*k])
    # number of remaining nodes if we add the next one to the current group
    m = n-k-1
    # if we don't have enough remaining nodes, merging this node means merging all of them 
    if m < k:
        c_merge = np.sum(np.array(group_degree)-np.array(degree_sequence[k:]))
    # otherwise the cost of merging is the cost of merging only this node + the cost of grouping the rest
    else:
        c_merge = group_degree-degree_sequence[k]+assignment_cost_additions_only(degree_sequence[k+1:2*k+1])
    # keep adding nodes while you can
    i = 0
    # while it's cheaper to merge the i-th node
    while c_new >= c_merge:
        # add i-th node to the current group
        # merge all nodes and return, if we don't have enough to proceed afterwards
        if m < k:
            anonymised_sequence = anonymised_sequence + [group_degree]*(m+1)
            return anonymised_sequence
        else:
            anonymised_sequence = anonymised_sequence + [group_degree]
        # checking the next node now
        i = i+1
        # number of remaining nodes after we added one to the current group
        m = n-k-1-i
        # cost of creating a new group starting from i
        c_new = assignment_cost_additions_only(degree_sequence[k+i:2*k+i])
        # if there are not enough nodes left to potentially form a new group
        if m < k:
            # cost of merging i and all the other nodes to the current group
            c_merge = np.sum(np.array(group_degree)-np.array(degree_sequence[k+i:]))
        # otherwise
        else:
            # cost of merging i and starting a new group from k+1
            c_merge = group_degree-degree_sequence[k+i]+assignment_cost_additions_only(degree_sequence[k+i+1:2*k+i+1])
    # when you stop adding new nodes, make a recursive call starting a new group on the remaining sequence
    anonymised_sequence = anonymised_sequence + greedy_degree_anonymiser(degree_sequence[k+i:],k)
    return anonymised_sequence

# Algorithm 1 in [1]
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
        # pick a random one
        idx = remaining_vertices[rn.randrange(len(remaining_vertices))]
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

# Section 6.1 in [1]
def find_max_swap(G,target_edges):
    edges = G.edges
    num_samples = int(np.floor(np.log(len(edges))))
    selected_edges = rn.sample(edges, k=num_samples) #change to k=num_samples for better performance but slower running time
    best_swap = (-1e8,None)
    for i in range(len(selected_edges)-1):
        for j in range(i+1,len(selected_edges)):
            e1 = selected_edges[i]
            e2 = selected_edges[j]
            if (e1[0],e2[0]) not in edges and (e1[1],e2[1]) not in edges:
                c = 0
                c = c - 1 if e1 in target_edges else c
                c = c - 1 if e2 in target_edges else c
                c = c + 1 if (e1[0],e2[0]) in target_edges else c
                c = c + 1 if (e1[1],e2[1]) in target_edges else c
                if c > best_swap[0]:
                    best_swap = (c,(e1,e2,(e1[0],e2[0]),(e1[1],e2[1])))
            if (e1[0],e2[1]) not in edges and (e1[1],e2[0]) not in edges:
                c = 0
                c = c - 1 if e1 in target_edges else c
                c = c - 1 if e2 in target_edges else c
                c = c + 1 if (e1[0],e2[1]) in target_edges else c
                c = c + 1 if (e1[1],e2[0]) in target_edges else c
                if c > best_swap[0]:
                    best_swap = (c,(e1,e2,(e1[0],e2[1]),(e1[1],e2[0])))
    return best_swap

# Section 6.1 in [1]
def greedy_swap(G, target_edges):
    result = find_max_swap(G,target_edges)
    if result[1] is None:
        print("Sorry, I couldn't find any good edge swap")
        return G
    else:
        (c, (e1, e2, ee1, ee2)) = result
    while c > 0:
        G.remove_edge(e1[0],e1[1])
        G.remove_edge(e2[0],e2[1])
        G.add_edge(ee1[0],ee1[1])
        G.add_edge(ee2[0],ee2[1])
        (c, (e1, e2, ee1, ee2)) = find_max_swap(G,target_edges)
    return G

# Section 6.2 in [1]
def priority(degree_sequence,target_edges):
    
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
        v = vd[idx][0]
                
        # iterate over all the degree-sorted vertices u such that (u,v) is an edge in the original graph
        for i,u in enumerate(vd):
            # stop when we added v_d edges
            if vd[idx][1] == 0:
                break
            # don't add self-loops
            if u[0] == v:
                continue
            # add the edge if this exists also in the original graph
            # we do an additional check for the degree of u to be > 0 because we may have skipped other edges not
            # belonging to G but that would have prevented us from using u and making its degree negative
            if (v,u[0]) in target_edges and u[1] > 0:
                G.add_edge(v,u[0])
                # decrease the degree of the connected vertex
                vd[i] = (u[0],u[1] - 1)
                # keep track of how many edges we added
                vd[idx] = (vd[idx][0],vd[idx][1] - 1)

        # iterate over all the degree-sorted vertices u such that (u,v) is NOT an edge in the original graph
        for i,u in enumerate(vd):
            # stop when we added v_d edges
            if vd[idx][1] == 0:
                break
            # don't add self-loops
            if u[0] == v:
                continue
            # now add edges that are not in the original graph
            if (v,u[0]) not in target_edges:
                G.add_edge(v,u[0])
                # decrease the degree of the connected vertex
                vd[i] = (u[0],u[1] - 1)
                # keep track of how many edges we added
                vd[idx] = (vd[idx][0],vd[idx][1] - 1)

# Anonymise G given a value of k. Uses Probing (Algorithm 2 in [1])
def graph_anonymiser(G,k,with_greedy_anonymisation=False,with_priority=True):
    degree_sequence = get_degree_sequence(G)
    degree_sequence.sort(reverse=True)
    if with_greedy_anonymisation:
         anonymised_sequence = greedy_degree_anonymiser(degree_sequence,k)
    else:
        anonymised_sequence = dp_degree_anonymiser(degree_sequence,k)
    if with_priority:
        Ga = priority(anonymised_sequence,G.edges())
    else:
        Ga = construct_graph(anonymised_sequence)
    while Ga is None:
        candidates = list(nx.non_edges(G))
        e = rn.sample(candidates, k=1)
        G.add_edge(e[0][0],e[0][1])
        degree_sequence = get_degree_sequence(G)
        degree_sequence.sort(reverse=True)
        if with_greedy_anonymisation:
            anonymised_sequence = greedy_degree_anonymiser(degree_sequence,k)
        else:
            anonymised_sequence = dp_degree_anonymiser(degree_sequence,k)
        if with_priority:
            Ga = priority(anonymised_sequence,G.edges())
        else:
            Ga = construct_graph(anonymised_sequence)
    return Ga