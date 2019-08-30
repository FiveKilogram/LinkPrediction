import networkx as nx
import math

def CN(graph_file):
    G = nx.read_edgelist(graph_file)
    ebunch = nx.non_edges(G)

    sim_dict = {}  # 存储相似度的字典
    i = 0

    for u, v in ebunch:
        i = i + 1
        s = len(list(nx.common_neighbors(G, u, v)))
        if (s > 0):
            sim_dict[(u, v)] = s
        print(i)
        print(str(u) + "," + str(v))

    print ("over")
    return sim_dict

CN('J:\\Python\\LinkPrediction\\Networks\\CE\\celegansneural.edgelist')
