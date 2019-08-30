# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:24:01 2018

@author: xsjxi
"""
import networkx as nx
import math
import weight_clustering3 as wc3

def pair(x, y):
    if (x < y):
        return (x, y)
    else:
        return (y, x)
    
def GWMI(G):
    
    edges = nx.edges(G)
    nodes = nx.nodes(G)
    beta = -math.log2(0.0001)
    sim_dict = {}
    g = nx.Graph()
    
    weight_dic = {}
    for u, v in edges:
        s = len(list(nx.common_neighbors(G, u, v)))
        weight_dic[(u, v)] = s + 1
        weight_dic[(v, u)] = s + 1
        g.add_edge(u, v, weight = weight_dic[(u, v)])          
    #print(weight_dic)
    all_weight = 0
    for u, v in edges:
        all_weight = all_weight + weight_dic[(u, v)]
    #print(all_weight)
    
    nodes_Weight_dict = {}
    weight_list = []
    
    # 得到每个点的“点权值”
    for v in nodes:
        node_weight = 0
        v_neighbors = nx.neighbors(G,v)
        for u in v_neighbors:
            node_weight += weight_dic[(u, v)]
        weight_list.append(node_weight)
        nodes_Weight_dict[v] = node_weight
    
    distinct_weight_list = list(set(weight_list))
    #print(nodes_Weight_dict)
    size = len(distinct_weight_list)
    
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
            for i in range(1, w_n + 1):
                p0 *= (b - i) / (a - i)
            if p0 == 1:
                self_Connect_dict[(w_n, w_m)] = beta
                self_Connect_dict[(w_m, w_n)] = beta
            else:
                self_Connect_dict[(w_n, w_m)] = -math.log2(1 - p0)
                self_Connect_dict[(w_m, w_n)] = -math.log2(1 - p0)
            #print (str(w_n) + "," + str(w_m))
            #print (self_Connect_dict[(w_n, w_m)])
    #print(self_Connect_dict)
    self_Conditional_dict = {}
    for z in nodes:
        w_z = nodes_Weight_dict[z]
        #d_z = nx.degree(G,z)
        if w_z > 1:
            alpha = 2 / (w_z * (w_z - 1))
            cc_z = wc3.weight_clustering3(g, z)#修改为加权聚类系数
            #cc_z = nx.clustering(G, z)
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
    #print(self_Conditional_dict)
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


# =============================================================================
# G = nx.Graph()
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
# GWMI(G)
# =============================================================================


