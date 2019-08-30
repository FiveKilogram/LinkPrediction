# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:00:35 2018

@author: xsjxi
"""
import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()
#G=nx.read_weighted_edgelist("test.edgelist")
#G = nx.read_weighted_edgelist('J:\\Python\\LinkPrediction\\test.edgelist')
G.add_weighted_edges_from([(1,4,2),(1,3,1),(1,2,2),(2,5,2),(2,4,1),(5,7,1),(5,6,1),(6,7,3),(6,8,1),(7,8,2)])
#G.add_weighted_edges_from([(1,4,1),(1,3,1),(1,2,1),(2,5,1),(2,4,1),(5,7,1),(5,6,1),(6,7,1),(6,8,1),(7,8,1)])

nodes = nx.nodes(G)
edges = nx.edges(G)
# =============================================================================
# for u, v in edges:
#     weight = G.get_edge_data(u,v)['weight']
#     print(str(u)+"------"+str(v)+"----"+str(weight))
# =============================================================================
for u in nodes:
    print(u)
    print(nx.clustering(G, u,'weight'))#加权聚类系数
    print(u)
    print(nx.clustering(G, u))#加权聚类系数
        
