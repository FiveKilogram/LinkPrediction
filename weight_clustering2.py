# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 22:03:04 2018

@author: xsjxi
"""
import networkx as nx
# =============================================================================
# G = nx.read_weighted_edgelist('J:\\release-flickr-links.txt')
# edges = nx.edges(G)
# for u, v in edges:
#     print(str(u)+"----"+str(v))
# =============================================================================
#网络科学导论第三种
def weight_clustering2(G,z):
    weight_cl = 0
    edges = nx.edges(G)
    edge_weight = {}
    edge_one_weight = {}
    max_weight = 0
    weight_cl = 0
    
    for u, v in edges:
        weight = G.get_edge_data(u,v)['weight']
        edge_weight[(u, v)] = weight
        edge_weight[(v, u)] = weight
        if weight >= max_weight:
            max_weight = weight
    for u, v in edges:
        edge_one_weight[(u, v)] = edge_weight[(u, v)]/max_weight
        edge_one_weight[(v, u)] = edge_weight[(u, v)]/max_weight
    cl1 = 0
    cl2 = 0
    z_degree = nx.degree(G, z)
    if z_degree == 1:
        return 0.0
    else:
        for u in nx.neighbors(G, z):
            for v in nx.neighbors(G, z):
                if u!=v: 
                    cl1 = cl1 + (edge_one_weight[u, z]*edge_one_weight[v, z])
                    if G.has_edge(u, v):
                        cl2 = cl2 + (edge_one_weight[(u, v)]*edge_one_weight[u, z]*edge_one_weight[v, z])
        weight_cl = cl2/cl1
    return weight_cl

#G = nx.read_weighted_edgelist('J:\\Python\\LinkPrediction\\celegansneural.edgelist')
G = nx.Graph()
#G.add_weighted_edges_from([(1,4,1),(1,3,1),(1,2,1),(2,5,1),(2,4,1),(5,7,1),(5,6,1),(6,7,1),(6,8,1),(7,8,1)])
#G.add_weighted_edges_from([(1,4,2),(1,3,1),(1,2,2),(2,5,2),(2,4,1),(5,7,1),(5,6,1),(6,7,3),(6,8,1),(7,8,2)])
#G.add_weighted_edges_from([(1,2,1),(1,3,0.001),(2,3,1)])
nodes = nx.nodes(G)
edges = nx.edges(G)

for u in nodes:
    print(u)
    print(weight_clustering(G,u))#加权聚类系数





















        
    

