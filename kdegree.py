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
    
# "degree anonymization cost" as defined in Section 8 of [1]
def assignment_cost_additions_deletions(degree_sequence):
    return np.sum(np.abs(np.int(np.median(degree_sequence))-np.array(degree_sequence)))
    
# Precomputation of the anonymisation cost, as described in Section 4 of [1]
def anonymisation_cost_precomputation(degree_sequence,k,with_deletions):
    n = np.size(degree_sequence)
    C = np.full([n,n],np.inf)
    for i in range(n-1):
        for j in range(i+k-1,np.min([i+2*k,n])):
            if with_deletions:
                C[i,j] = assignment_cost_additions_deletions(degree_sequence[i:j+1])
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
    cost, anonymised_sequence = dp_degree_anonymiser_recursion(degree_sequence,k,C,n,with_deletions,0,Da,sequences)
    return anonymised_sequence

# The dynamic programming algorithm described in Section 4 of [1] - recursion part
def dp_degree_anonymiser_recursion(degree_sequence,k,C,n,with_deletions,i,Da,sequences):
    
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
        iteration = 1
        for t in range(np.max([k-1,n-2*k]),n-k):
            iteration = iteration + 1
            if Da[t] == np.inf:
                cost, sequence = dp_degree_anonymiser_recursion(degree_sequence[0:t+1],k,C,t+1,with_deletions,i+1,Da,sequences)
                Da[t] = cost
                sequences[t] = sequence
            else:
                cost = Da[t]
                sequence = sequences[t]
            cost = cost + C[t+1,n-1]
            if cost < min_cost:
                min_cost = cost
                min_cost_sequence = np.concatenate((sequence,np.full(np.size(degree_sequence[t+1:]),degree_sequence[t+1])))                
            #if with_deletions:
                #sequences.append(sequence + [np.int(np.median(degree_sequence[t+1:end_idx-1]))]*len(degree_sequence[t+1:]))
            #else:
                #sequences.append(np.concatenate((sequence,np.full(np.size(degree_sequence[t+1:]),degree_sequence[t+1]))))
                # sequences.append(sequence + [degree_sequence[t+1]]*len(degree_sequence[t+1:]))
        to_return = (min_cost, min_cost_sequence) if min_cost < all_in_one_group_cost else (all_in_one_group_cost, all_in_one_group_sequence)
        return to_return

# The greedy algorithm described in Section 4 of [1]
def greedy_degree_anonymiser(degree_sequence,k,with_deletions=False):
    # create a k-anonymomus group with all nodes having degree equal to the largest in the group
    # degree_sequence is sorted, so the largest element is the first one
    group_degree = degree_sequence[0]
    n = np.size(degree_sequence)
        
    # if there are less than 2*k nodes left to anonymise, group them together and finish
    if n < 2*k:
        #if with_deletions:
        #    anonymised_sequence = [np.int(np.median(degree_sequence))]*n
        #else:
        #    anonymised_sequence = [group_degree]*n
        anonymised_sequence = np.full(n,group_degree)
        return anonymised_sequence
    # otherwise start a new group
    #if with_deletions:
    #    anonymised_sequence = [np.int(np.median(degree_sequence[0:k]))]*k
    #else:
    #    anonymised_sequence = [group_degree]*k
    anonymised_sequence = np.full(k,group_degree)
    # then check if we can add another node to the current group or if it's better to create a new group
    # cost of grouping the next k nodes together (we know we have at least k nodes left, thanks to the IF above)
    #c_new = assignment_cost_additions_deletions(degree_sequence[k:2*k]) if with_deletions else assignment_cost_additions_only(degree_sequence[k:2*k])
    c_new = assignment_cost_additions_only(degree_sequence[k:2*k])
    # number of remaining nodes if we add the next one to the current group
    m = n-k-1
    # if we don't have enough remaining nodes, merging this node means merging all of them 
    if m < k:
        #if with_deletions:
        #    c_merge = assignment_cost_additions_deletions(degree_sequence[k:])
        #else:
        #    c_merge = np.sum(np.array(group_degree)-np.array(degree_sequence[k:]))
        c_merge = np.sum(group_degree-degree_sequence[k:])
    # otherwise the cost of merging is the cost of merging only this node + the cost of grouping the rest
    else:
        #if with_deletions:
        #    c_merge = assignment_cost_additions_deletions(degree_sequence[k:2*k+1])
        #else:
        #    c_merge = group_degree-degree_sequence[k]+assignment_cost_additions_only(degree_sequence[k+1:2*k+1])
        c_merge = group_degree-degree_sequence[k]+assignment_cost_additions_only(degree_sequence[k+1:2*k+1])
    # keep adding nodes while you can
    i = 0
    # while it's cheaper to merge the i-th node
    while c_new >= c_merge:
        # add i-th node to the current group
        # merge all nodes and return, if we don't have enough to proceed afterwards
        if m < k:
            #if with_deletions:
            #    anonymised_sequence = anonymised_sequence + [np.int(np.median(degree_sequence[0:k+i]))]*(m+1)
            #else:
            #    anonymised_sequence = anonymised_sequence + [group_degree]*(m+1)
            anonymised_sequence = np.concatenate((anonymised_sequence,np.full(m+1,group_degree)))
            return anonymised_sequence
        else:
            #if with_deletions:
            #    anonymised_sequence = anonymised_sequence + [np.int(np.median(degree_sequence[0:k+i]))]
            #else:
            #    anonymised_sequence = anonymised_sequence + [group_degree]
            anonymised_sequence = np.concatenate((anonymised_sequence,np.full(1,group_degree)))
        # checking the next node now
        i = i+1
        # number of remaining nodes after we added one to the current group
        m = n-k-1-i
        # cost of creating a new group starting from i
        #c_new = assignment_cost_additions_deletions(degree_sequence[k+i:2*k+i]) if with_deletions else assignment_cost_additions_only(degree_sequence[k+i:2*k+i])
        c_new = assignment_cost_additions_only(degree_sequence[k+i:2*k+i])
        # if there are not enough nodes left to potentially form a new group
        if m < k:
            # cost of merging i and all the other nodes to the current group
            #if with_deletions:
            #    c_merge = assignment_cost_additions_deletions(degree_sequence[k+i:])
            #else:
            #    c_merge = np.sum(np.array(group_degree)-np.array(degree_sequence[k+i:]))
            c_merge = np.sum(group_degree-degree_sequence[k+i:])
        # otherwise
        else:
            # cost of merging i and starting a new group from k+1
            #if with_deletions:
            #    c_merge = assignment_cost_additions_deletions(degree_sequence[k+i:2*k+i+1])
            #else:
            #    c_merge = group_degree-degree_sequence[k+i]+assignment_cost_additions_only(degree_sequence[k+i+1:2*k+i+1])
            c_merge = group_degree-degree_sequence[k+i]+assignment_cost_additions_only(degree_sequence[k+i+1:2*k+i+1])
    # when you stop adding new nodes, make a recursive call starting a new group on the remaining sequence
    #anonymised_sequence = anonymised_sequence + greedy_degree_anonymiser(degree_sequence[k+i:],k,with_deletions=with_deletions)
    anonymised_sequence = np.concatenate((anonymised_sequence,greedy_degree_anonymiser(degree_sequence[k+i:],k,with_deletions=with_deletions)))
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
def find_max_swap(G,original_G,already_sampled):
    edges = set(G.edges())
    edges = edges.difference(already_sampled)
    num_edges = len(edges)
    num_samples = int(np.floor(num_edges*0.2))
    print(num_edges,num_samples)
    selected_edges = rn.sample(edges, k=num_samples)
    #num_samples = int(np.floor(np.log(len(edges))))
    #selected_edges = rn.sample(edges, k=np.min([60*num_samples,len(edges)])) #k=num_samples fast but bad, k=len(edges) slow but good
    best_swap = (-1e8,None)
    
    for i in range(len(selected_edges)-1):
        for j in range(i+1,len(selected_edges)):
            e1 = selected_edges[i]
            e2 = selected_edges[j]
            
            c = 0
            c = c - 1 if original_G.has_edge(e1[0],e1[1]) else c
            c = c - 1 if original_G.has_edge(e2[0],e2[1]) else c
            
            # I don't think this is working, but the idea is that: only consider swaps where there is some potential overlap with original_G
            if c == -2:
                already_sampled.add(e1)
                already_sampled.add(e2)
                continue
            
            if not G.has_edge(e1[0],e2[0]) and not G.has_edge(e1[1],e2[1]):
                c = c + 1 if original_G.has_edge(e1[0],e2[0]) else c
                c = c + 1 if original_G.has_edge(e1[1],e2[1]) else c
                if c > best_swap[0]:
                    best_swap = (c,(e1,e2,(e1[0],e2[0]),(e1[1],e2[1])))
                    
            if not G.has_edge(e1[0],e2[1]) and not G.has_edge(e1[1],e2[0]):
                c = 0
                c = c + 1 if original_G.has_edge(e1[0],e2[1]) else c
                c = c + 1 if original_G.has_edge(e1[1],e2[0]) else c
                if c > best_swap[0]:
                    best_swap = (c,(e1,e2,(e1[0],e2[1]),(e1[1],e2[0])))
                    
            # no swap can lead to a better improvement in edges overlap
            if best_swap[0] == 2:
                return best_swap, already_sampled
    return best_swap, already_sampled

# Section 6.1 in [1]
def greedy_swap(G, original_G):
    
    already_sampled = set()
    result, already_sampled = find_max_swap(G,original_G, already_sampled)
    
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
        
        #already_sampled.add(ee1)
        #already_sampled.add(ee2)

        (c, (e1, e2, ee1, ee2)), already_sampled = find_max_swap(G,original_G,already_sampled)
    return G

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

# Anonymise G given a value of k. Uses Probing (Algorithm 2 in [1])
# Choices for algorithm:
# 0: dp (greedy) + supergraph -> ADDITIONS ONLY
# 1: dp (greedy) + construct graph + greedy swap -> ADDITIONS AND SWAPS
# 2: dp (greedy) + priority -> ADDITIONS AND SWAPS (priority version - faster)
# 3: dp2 (greedy2) + construct graph + greedy swap -> ADDITIONS, SWAPS, AND DELETIONS
def graph_anonymiser(G,k,algorithm=0,noise=1,greedy_anonymiser=True):
    dv = [(d,v) for v, d in G.degree()]
    degree_sequence,permutation = sort_dv(dv)
        
    if algorithm == 0:
        print("Not implemented yet")
        return -1
        
    elif algorithm == 1:
        attempt = 1
        print("Attempt number",attempt)
        anonymised_sequence = greedy_degree_anonymiser(degree_sequence,k,with_deletions=False) \
            if greedy_anonymiser else dp_degree_anonymiser(degree_sequence,k,with_deletions=False)
        new_anonymised_sequence = [None] * len(degree_sequence)
        for i in range(len(permutation)):
            new_anonymised_sequence[permutation[i]] = anonymised_sequence[i]
        anonymised_sequence = new_anonymised_sequence
        
        Ga = construct_graph(anonymised_sequence)
        
        while Ga is None:
            attempt = attempt+1
            print("Attempt number",attempt)
            
            dv = probing(dv,noise)
            degree_sequence,permutation = sort_dv(dv)
            
            anonymised_sequence = greedy_degree_anonymiser(degree_sequence,k,with_deletions=False) \
                if greedy_anonymiser else dp_degree_anonymiser(degree_sequence,k,with_deletions=False)
            new_anonymised_sequence = [None] * len(degree_sequence)
            for i in range(len(permutation)):
                new_anonymised_sequence[permutation[i]] = anonymised_sequence[i]
            anonymised_sequence = new_anonymised_sequence
            
            if not nx.is_valid_degree_sequence_erdos_gallai(anonymised_sequence):
                continue
            Ga = construct_graph(anonymised_sequence)
            if Ga is None:
                print("the sequence is valid but the graph construction failed")
                
        Ga = greedy_swap(Ga,G)
        
    elif algorithm == 2:
        attempt = 1
        print("Attempt number",attempt)
        anonymised_sequence = greedy_degree_anonymiser(degree_sequence,k,with_deletions=False) \
            if greedy_anonymiser else dp_degree_anonymiser(degree_sequence,k,with_deletions=False)
        
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
            
            anonymised_sequence = greedy_degree_anonymiser(degree_sequence,k,with_deletions=False) \
                if greedy_anonymiser else dp_degree_anonymiser(degree_sequence,k,with_deletions=False)
            
            new_anonymised_sequence = [None] * len(degree_sequence)
            for i in range(len(permutation)):
                new_anonymised_sequence[permutation[i]] = anonymised_sequence[i]
            anonymised_sequence = new_anonymised_sequence
            
            if not nx.is_valid_degree_sequence_erdos_gallai(anonymised_sequence):
                continue
            Ga = priority(anonymised_sequence,G)
            if Ga is None:
                print("the sequence is valid but the graph construction failed")
            
    elif algorithm == 3:
        print("Not implemented yet")
        return -1
        '''
        anonymised_sequence = greedy_degree_anonymiser(degree_sequence,k,with_deletions=True) \
            if greedy_anonymiser else dp_degree_anonymiser(degree_sequence,k,with_deletions=True)
        Ga = construct_graph(anonymised_sequence)
        Ga = greedy_swap(Ga,G.edges())
        '''
        
    else:
        print("algorithm should be a number between 0 and 3. 0 [add], 1 [add/swap], 2 [add/swap w priority], 3 [add/swap/delete]")
        return -1
    return Ga