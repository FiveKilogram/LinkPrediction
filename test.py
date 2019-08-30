import networkx as nx
import sim2

net_file = 'F:\\OneDrive\\LinkPrediction\\Networks\\test\\test.edgelist'
sim_method = 'HCR'

print(sim_method)

print(net_file)

G = nx.read_edgelist(net_file, nodetype=int)
#G = G.to_undirected()
#G = nx.convert_node_labels_to_integers(G)

sim_dict = sim2.similarities(G, sim_method)

for (u, v) in sim_dict.keys():
    print('(%d, %d): %f' % (u, v, sim_dict[(u, v)]))
#end for
