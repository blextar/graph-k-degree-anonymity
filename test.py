import networkx as nx
import numpy as np
import random as rn
import kdegree as kd
import time
import sys
import getopt

def main(argv):
  
  input_file = None
  k = None
  noise = None

  try:
    opts, args = getopt.getopt(argv,"hi:k:n:")
  except getopt.GetoptError:
    print('test.py -i <inputfile> -k <k> -n <noise>')
    sys.exit(2)
  for opt, arg in opts:
    print(opt,arg) 
    if opt == '-h':
      print('test.py -i <inputfile> -k <k> -n <noise>')
      sys.exit()
    elif opt is "-i":
      input_file = arg
    elif opt is "-k":
      k = arg
    elif opt is "-n": 
      noise = arg 
  
  
  G = nx.read_edgelist(input_file,nodetype=int)
  
  start = time.time()
  Ga = kd.graph_anonymiser(G,k=k,algorithm=2,noise=noise,greedy_anonymiser=False)
  print("Total execution time =",time.time()-start)

  H = nx.intersection(G,Ga)

  num_edges_in_G = len(set(G.edges()))
  num_edges_in_both = len(set(H.edges()))

  print("Edges overlap = " + str(100*num_edges_in_both/num_edges_in_G)+"%")
  print("Num edges original graph = " + str(nx.number_of_edges(G)))
  print("Num edges anonymised graph = " + str(nx.number_of_edges(Ga)))

if __name__ == "__main__":
   main(sys.argv[1:])
