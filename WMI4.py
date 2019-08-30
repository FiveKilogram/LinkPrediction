# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:01:41 2018

@author: xsjxi
"""

#coding:utf-8
import networkx as nx
import math
import matplotlib.pyplot as plt

def pair(x, y):
    if (x < y):
        return (x, y)
    else:
        return (y, x)
# 把MI方法在加权图上进行拓展，对计算公式进行加权考虑
        
def WMI(G):
    #G = nx.read_edgelist(graph_file)
    
    edges = nx.edges(G)
    nodes = nx.nodes(G)
    beta = -math.log2(0.0001)
    sim_dict = {}
    
    # 得到图中所有边的权值之和
    all_weight = 0
    for u, v in edges:
        all_weight = all_weight + G.get_edge_data(u,v)['weight']
    print(all_weight)
        
    # 计算图中不同‘点权值’的点之间相连的互信息
    nodes_Weight_dict = {}
    weight_list = []
    
    # 得到每个点的“点权值”
    for v in nodes:
        node_weight = 0
        v_neighbors = nx.neighbors(G,v)
        for u in v_neighbors:
            node_weight += G.get_edge_data(u,v)['weight']
        weight_list.append(node_weight)
        nodes_Weight_dict[v] = node_weight
    #print(weight_list)
    #print(nodes_Weight_dict)
          
    distinct_weight_list = list(set(weight_list))
    #print(distinct_weight_list)
    size = len(distinct_weight_list)
    #print(size)
    
    self_Connect_dict = {}
    #得到不同‘点权值’的点之间相连的互信息
    for x in range(size):
        w_x = distinct_weight_list[x]
        for y in range(x, size):
            w_y = distinct_weight_list[y]
            p0 = 1
            (w_n, w_m) = pair(w_x, w_y)
            a = all_weight + 1
            b = all_weight - w_m + 1
            for i in range(1, int(w_n + 1)):
                p0 *= (b - i) / (a - i)
            if p0 == 1:
                self_Connect_dict[(w_n, w_m)] = beta
                #self_Connect_dict[(w_m, w_n)] = beta
            else:
                self_Connect_dict[(w_n, w_m)] = -math.log2(1 - p0)
                #self_Connect_dict[(w_m, w_n)] = -math.log2(1 - p0)
            #print (str(w_n) + "," + str(w_m))
            #print (self_Connect_dict[(w_n, w_m)])
    #print(self_Connect_dict)
    self_Conditional_dict = {}
    for z in nodes:
        w_z = nodes_Weight_dict[z]
        if w_z > 1:
            alpha = 2 / (w_z * (w_z - 1))
            cc_z = nx.clustering(G, z)#修改为加权聚类系数
            if cc_z == 0:
                log_c = beta
            else:
                log_c = -math.log2(cc_z)
            # end if
            s = 0
            neighbor_list = nx.neighbors(G,z)
            size = len(neighbor_list)
            for i in range(size):
                m = neighbor_list[i]
                for j in range(i+1,size):
                    n = neighbor_list[j]
                    (k_x, k_y) = pair(nodes_Weight_dict[m], nodes_Weight_dict[n])
                    if i!=j:
                        s += (self_Connect_dict[(k_x, k_y)] - log_c)
            self_Conditional_dict[z] = alpha * s
    print(self_Conditional_dict)
    sim_dict = {}  # 存储相似度的字典
    ebunch = nx.non_edges(G)
    
    for x, y in ebunch:
        s = 0
        (k_x, k_y) = pair(nodes_Weight_dict[x], nodes_Weight_dict[y])
        for z in nx.common_neighbors(G, x, y):
            s += self_Conditional_dict[z]
        sim_dict[(x, y)] = s - self_Connect_dict[(k_x, k_y)]
        # end if
    # end for
    #print(sim_dict)
    return sim_dict


#G = nx.Graph()
#G.add_weighted_edges_from([(1,4,1),(1,3,1),(1,2,1),(2,5,1),(2,4,1),(5,7,1),(5,6,1),(6,7,1),(6,8,1),(7,8,1)])
G = nx.read_weighted_edgelist('J:\\Python\\LinkPrediction\\test.edgelist',nodetype=int)
WMI(G)


