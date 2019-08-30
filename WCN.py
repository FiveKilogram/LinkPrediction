# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:46:30 2018

@author: xsjxi
"""
import networkx as nx
import matplotlib.pyplot as plt

def weight_common_neighbors_index(G):
    #print("one time")

    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    # print (node_num)
    # print (edge_num)

    ebunch = nx.non_edges(G)
    
    sim_dict = {}   # 存储相似度的字典
    
    
    for u, v in ebunch:
        edge_weight = 0
        for z in nx.common_neighbors(G,u,v):
            edge_weight = edge_weight + G.get_edge_data(u,z)['weight'] +G.get_edge_data(v,z)['weight']        
        if (edge_weight > 0):
            sim_dict[(u, v)] = edge_weight
    return sim_dict