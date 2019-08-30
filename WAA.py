# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:49:05 2018

@author: xsjxi
"""
import networkx as nx
import math
import matplotlib.pyplot as plt

def weight_adamic_adar_index(G):
    ebunch = nx.non_edges(G)
    nodes  = nx.nodes(G)
    
    sim_dict = {}   # 存储相似度的字典
    weight_dic = {}
    
    
    for m in nodes:
        l = 0
        for n in nx.neighbors(G, m):
            l = l + G.get_edge_data(m,n)['weight']
        weight_dic[m] = l
    
    for u, v in ebunch:
        s = 0
        for z in nx.common_neighbors(G,u,v):
            s = s + (G.get_edge_data(u,z)['weight'] + G.get_edge_data(v,z)['weight'])/math.log2(1 + weight_dic[z])
        if (s > 0):
            sim_dict[(u, v)] = s
    return sim_dict