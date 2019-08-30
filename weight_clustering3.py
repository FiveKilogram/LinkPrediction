# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 16:18:23 2018

@author: xsjxi
"""
import networkx as nx
#网络科学导论第一种
def weight_clustering3(G,z):
    average_weight = 0
    weight = 0
    for u in nx.neighbors(G, z):
        weight = weight + G.get_edge_data(u,z)['weight']
    degree = nx.degree(G, z)
    average_weight = weight/degree
    si = degree*average_weight
    
    
    cl = 0
    z_degree = nx.degree(G, z)
    weight_cl = 0
    
    if z_degree == 1:
        return 0.0
    else:
        for u in nx.neighbors(G, z):
            u_weight = G.get_edge_data(u,z)['weight']
            for v in nx.neighbors(G, z):
                v_weight = G.get_edge_data(v,z)['weight']
                if (u!=v)&(G.has_edge(u, v)):
                    cl = cl + (u_weight + v_weight)/2
        weight_cl = (1/(si*(degree-1)))*cl
    return weight_cl