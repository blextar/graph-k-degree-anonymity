import networkx as nx
import kdegree as kd
import time
import sys
import getopt
from os import path

def main(argv):
  
    input_file = None
    output_file = None
    k = None
    noise = None

    try:
        opts, args = getopt.getopt(argv,"i:o:k:n:")
    except getopt.GetoptError:
        sys.exit("test.py -i <inputfile> -o <outputfile> -k <k-anonymity level> -n <noise>")
    for opt, arg in opts:
        if opt == '-i':
            input_file = arg
        elif opt == '-o':
            output_file = arg
        elif opt == '-k':
            k = int(arg)
        elif opt == '-n': 
            noise = int(arg)
            
    error = False
    if input_file is None:
        print("Please specify an input file")
        error = True
    if output_file is None:
        print("Please specify an output file")
        error = True
    if error:
        sys.exit("Syntax: test.py -i <inputfile> -o <outputfile> -k <k-anonymity level> -n <noise>")
        
    if k is None:
        k = 2
        print("Using default k = 2")
    if noise is None:
        noise = 1
        print("Using default n = 1")

    if not path.exists(input_file):
        sys.exit("Cannot find the input file")
    
    log = open(output_file + '.log','w')
    sys.stdout = log
    
    G = nx.read_edgelist(input_file,nodetype=int)
            
    start = time.time()
    Ga = kd.graph_anonymiser(G,k=k,noise=noise,with_deletions=True)
    print("Total execution time =",time.time()-start)

    H = nx.intersection(G,Ga)

    num_edges_in_G = len(set(G.edges()))
    num_edges_in_both = len(set(H.edges()))

    print("Edges overlap = " + str(100*num_edges_in_both/num_edges_in_G)+"%")
    print("Num edges original graph = " + str(nx.number_of_edges(G)))
    print("Num edges anonymised graph = " + str(nx.number_of_edges(Ga)))
   
    nx.write_edgelist(Ga,output_file,data=False)
    
if __name__ == "__main__":
    main(sys.argv[1:])
