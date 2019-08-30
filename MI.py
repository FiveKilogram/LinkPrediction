# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 20:27:37 2017

@author: xsjxi
"""
#coding:utf-8
import networkx as nx
import math
from scipy.special import comb#组合方法
import matplotlib.pyplot as plt

#MI方法
def MI(graph_file):
    G = nx.read_edgelist(graph_file)
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    print (node_num)
    print (edge_num)
    sim_dict = {}  # 存储相似度的字典
    i = 0

    ebunch = nx.non_edges(G)

    for u, v in ebunch:
        pConnect = 0
        pMutual_Information = 0
        uDegree = nx.degree(G, u)
        vDegree = nx.degree(G, v)

        pConnect = 1 - (comb((edge_num - uDegree),vDegree))/(comb(edge_num,vDegree))
        I_pConnect = -math.log2(pConnect)

        for z in nx.common_neighbors(G, u, v):
            neighbor_num = len(list(nx.neighbors(G,z)))
            neighbor_list = nx.neighbors(G,z)
            for m  in range(len(neighbor_list)):
                for n in range(m+1,len(neighbor_list)):
                    if m!=n:
                        mDegree = nx.degree(G, neighbor_list[m])
                        nDegree = nx.degree(G, neighbor_list[n])
                        ppConnect = 1 - (comb((edge_num - mDegree),nDegree))/(comb(edge_num,nDegree))
                        I_ppConnect = -math.log2(ppConnect)
                        if nx.clustering(G,z) == 0:
                            pMutual_Information = pMutual_Information + (2 / (neighbor_num * (neighbor_num - 1))) * ((I_ppConnect) - (-math.log2(0.0001)))
                        else:
                            pMutual_Information = pMutual_Information + ( 2/ (neighbor_num * (neighbor_num - 1))) * ((I_ppConnect)-(-math.log2(nx.clustering(G,z))))
            #print(neighbor_num)
            #pMutual_Information = 0
        sim_dict[(u, v)] = -(I_pConnect - pMutual_Information)
        i = i + 1
        print(i)
        print(str(u) + "," + str(v))
        print(sim_dict[(u, v)])
    return sim_dict

MI('J:\\Python\\LinkPrediction\\Networks\\CE\\celegansneural.edgelist')
#G = nx.Graph()
# G.add_edge(1,2)
# G.add_edge(2,5)
# G.add_edge(2,3)
# G.add_edge(2,4)
# G.add_edge(3,6)
# G.add_edge(5,6)
# G.add_edge(4,6)

# G.add_edge(1,2)
# G.add_edge(1,3)
# G.add_edge(1,4)
# G.add_edge(2,4)
# G.add_edge(2,5)
# G.add_edge(5,6)
# G.add_edge(5,7)
# G.add_edge(6,7)
# G.add_edge(6,8)
# G.add_edge(7,8)
#MI(G)
# nx.draw_networkx(G)
# plt.show()

