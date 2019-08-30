# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:09:22 2018

@author: xsjxi
"""
import networkx as nx
#此种方法和networkx中默认的加权聚类系数一样
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

    z_degree = nx.degree(G, z)
    
    cl = 0
    if z_degree == 1:
        return 0.0
    else:
        for u in nx.neighbors(G, z):
            for v in nx.neighbors(G, z):
                if (u!=v)&(G.has_edge(u, v)):
                    cl = cl + (edge_one_weight[(u, v)]*edge_one_weight[u, z]*edge_one_weight[v, z])**(1/3)
        weight_cl = (1/(z_degree*(z_degree - 1)))*cl
    return weight_cl


# =============================================================================
# #G = nx.read_weighted_edgelist('J:\\Python\\LinkPrediction\\celegansneural.edgelist')
# G = nx.Graph()
# #G.add_weighted_edges_from([(1,4,1),(1,3,1),(1,2,1),(2,5,1),(2,4,1),(5,7,1),(5,6,1),(6,7,1),(6,8,1),(7,8,1)])
# G.add_weighted_edges_from([(1,4,2),(1,3,1),(1,2,2),(2,5,2),(2,4,1),(5,7,1),(5,6,1),(6,7,3),(6,8,1),(7,8,2)])
# nodes = nx.nodes(G)
# edges = nx.edges(G)
# 
# for u in nodes:
#     print(u)
#     print(weight_clustering(G,u))#加权聚类系数
# =============================================================================
    
    
    
    
    
    
# =============================================================================
# G = nx.Graph()
# G.add_weighted_edges_from([(1,2,2.0),(1,3,1.0),(2,4,2.0),(2,5,2.0),(5,6,1.0),(5,7,8.0),(6,7,1.0),(6,8,3.0),(7,8,2.0)])
# edges = nx.edges(G)
# nodes = nx.nodes(G)
# for v in nodes:
#     cc_z = nx.clustering(G, v)
#     #gt = nx.triangles(G,v)
#     print(str(v)+"-----"+str(cc_z))
#     #print(str(v)+"-----"+str(gt))
# print(nx.dijkstra_path(G, 3, 6))
#     
# =============================================================================
    