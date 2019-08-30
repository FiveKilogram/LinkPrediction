# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 20:27:37 2017

@author: xsjxi
"""
#coding:utf-8
import networkx as nx
import math
from scipy.special import comb#��Ϸ���
import matplotlib.pyplot as plt
import warnings

#MI����
def MI(graph_file):
    G = nx.read_edgelist(graph_file)
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    print (node_num)
    print (edge_num)
    sim_dict = {}  # �洢���ƶȵ��ֵ�
    i = 0

    I_pConnect_dict = {}
    edges = nx.edges(G)
    ebunch = nx.non_edges(G)
    

    for u, v in edges:
        uDegree = nx.degree(G, u)
        vDegree = nx.degree(G, v)
        pConnect = 1 - (comb((edge_num - uDegree), vDegree)) / (comb(edge_num, vDegree))
        I_pConnect = -math.log2(pConnect)
        I_pConnect_dict[(u, v)] = I_pConnect
        I_pConnect_dict[(v, u)] = I_pConnect
    for u, v in ebunch:
        uDegree = nx.degree(G, u)
        vDegree = nx.degree(G, v)
        pConnect = 1 - (comb((edge_num - uDegree), vDegree)) / (comb(edge_num, vDegree))
        I_pConnect = -math.log2(pConnect)
        I_pConnect_dict[(u, v)] = I_pConnect
        I_pConnect_dict[(v, u)] = I_pConnect


    ebunchs = nx.non_edges(G)
    i = 0
    for u, v in ebunchs:

        pMutual_Information = 0
        I_pConnect = I_pConnect_dict[(u, v)]
        for z in nx.common_neighbors(G, u, v):
            neighbor_num = len(list(nx.neighbors(G,z)))
            neighbor_list = nx.neighbors(G,z)
            for m  in range(len(neighbor_list)):
                for n in range(m+1,len(neighbor_list)):
                    if m!=n:
                        I_ppConnect = I_pConnect_dict[(neighbor_list[m], neighbor_list[n])]
                        if nx.clustering(G,z) == 0:
                            pMutual_Information = pMutual_Information + (2 / (neighbor_num * (neighbor_num - 1))) * ((I_ppConnect) - (-math.log2(0.0001)))
                        else:
                            pMutual_Information = pMutual_Information + ( 2/ (neighbor_num * (neighbor_num - 1))) * ((I_ppConnect)-(-math.log2(nx.clustering(G,z))))
        sim_dict[(u, v)] = -(I_pConnect - pMutual_Information)
        i = i + 1
# =============================================================================
#         print(i)
#         print(str(u) + "," + str(v))
#         print (sim_dict[(u, v)])
# =============================================================================
    print(sim_dict)
    return sim_dict

MI('J:\\Python\\LinkPrediction\\Networks\\test\\test.edgelist')