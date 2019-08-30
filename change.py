import networkx as nx

G = nx.read_pajek('.\\Networks\\SciMet\\SciMet.paj',type == int)
nx.write_edgelist(G, '.\\Networks\\SciMet\\SciMet.edgelist',type == int)