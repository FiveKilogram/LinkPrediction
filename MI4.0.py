# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 20:27:37 2017

@author: xsjxi
"""
#coding:utf-8
import networkx as nx
import math
#from scipy.special import comb#��Ϸ���
import matplotlib.pyplot as plt
#import warnings

#MI����
def MI(graph_file):
    G = nx.read_edgelist(graph_file)
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    print (node_num)
    print (edge_num)
    sim_dict = {}  # �洢���ƶȵ��ֵ�

    I_pConnect_dict = {}
    pDisConnect = 1
    edges = nx.edges(G)
    ebunch = nx.non_edges(G)
    nodes = nx.nodes(G)
    nodes_Degree_dict = {}
    for v in nodes:
        nodes_Degree_dict[v] = nx.degree(G, v)
        
    # 需要经常获取顶点的度，因此，可以事先存储下来
    # degree_list = [nx.degree(G, v) for v in range(G.number_of_nodes())]

    # 下面的两个循环计算$P(L^1_{xy})$，其实我们只需要计算不同度的值$P(L^1_{kxky})$
# =============================================================================
#     degree_I_pConnect = {}
#     for u, v in edges:
# =============================================================================
        
    for u, v in edges: 
        uDegree = nodes_Degree_dict[u]
        vDegree = nodes_Degree_dict[v]
        for i in range(1,vDegree + 1):
            pDisConnect = pDisConnect * (((edge_num - uDegree) - i + 1) / (edge_num - i + 1))
        pConnect = 1 - pDisConnect
        if pConnect == 0:
            I_pConnect = -math.log2(0.0001)
        else:
            I_pConnect = -math.log2(pConnect)
        I_pConnect_dict[(u, v)] = I_pConnect
        I_pConnect_dict[(v, u)] = I_pConnect
        pDisConnect = 1
    
    for m, n in ebunch:
# =============================================================================
#         mDegree = nx.degree(G, m)
#         nDegree = nx.degree(G, n)
# =============================================================================
        mDegree = nodes_Degree_dict[m]
        nDegree = nodes_Degree_dict[n]
        for i in range(1,nDegree + 1):
            pDisConnect = pDisConnect * (((edge_num - mDegree) - i + 1) / (edge_num - i + 1))
        pConnect = 1 - pDisConnect    
        if pConnect == 0:
            I_pConnect = -math.log2(0.0001)
        else:
            I_pConnect = -math.log2(pConnect)
        I_pConnect_dict[(m, n)] = I_pConnect
        I_pConnect_dict[(n, m)] = I_pConnect
        pDisConnect = 1

    ebunchs = nx.non_edges(G)
    i = 0

    # $I(L^1_{xy};z) = I(L^1;z)$，与x, y没有关系，可以先计算出来
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
        #print(i)
        print(str(u) + "," + str(v))
        print (sim_dict[(u, v)])
    return sim_dict

#MI('J:\\Python\\LinkPrediction\\Networks\\PB\\polblogs.edgelist')
#MI('J:\\Python\\LinkPrediction\\Networks\\CE\\celegansneural.edgelist')
#MI('J:\\Python\\LinkPrediction\\Networks\\jazz\\jazz.edgelist')
#MI('J:\\Python\\LinkPrediction\\Networks\\Hamster\\Hamster.edgelist')
MI('J:\\Python\\LinkPrediction\\Networks\\test\\test.edgelist')


