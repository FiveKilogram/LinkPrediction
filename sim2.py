# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:57:47 2016

@author: Longjie Li
"""

import networkx as nx
import numpy as np
import scipy.stats.stats as stats
import math
from scipy.special import comb
import weight_clustering3 as wc3

#计算顶点间的相似度
#@param graph 训练网络
#@param method 计算相似度的方法
#@param p1 相似度计算方法中的参数1
#@param p2 相似度计算方法中的参数2
def similarities(graph, method):
    method = method.upper()
    
    # 几个常用的方法    
    if method == 'CN':
        return common_neighbors_index(graph)
    if method == 'RA':
        return resource_allocation_index(graph)
    if method == 'LNB_CN':
        return LNB(graph, 'CN')
    if method == 'LNB_AA':
        return LNB(graph, 'AA')
    if method == 'LNB_RA':
        return LNB(graph, 'RA')
    if method == 'CLNB_CN':
        return CLNB(graph, 'CN')
    if method == 'CLNB2_CN':
        return CLNB2(graph, 'CN')
    if method == 'CLNB2_AA':
        return CLNB2(graph, 'AA')
    if method == 'CLNB2_RA':
        return CLNB2(graph, 'RA')
    if method == 'JACCARD':
        return jaccard_coefficient(graph)
    if method == 'AA':
        return adamic_adar_index(graph)
    if method == 'PA':
        return preferential_attachment_index(graph)
    if method == 'ADP':
        return adaptive_degree_penalization(graph, 2.5)
    if method == 'CCLP':
        return CCLP(graph)
    if method == 'NLC':
        return NLC(graph)
    if method == 'LCCL':
        return local_path_index(graph, 1)

    if method == 'ERA':
        return extend_resource_allocation_index(graph, 0.04)
    if method == 'CN_PA':
        return CN_PA(graph, 0.001)
    if method == 'LP':
        return local_path_index(graph, 0.001)
    if method == 'CNAD':
        return common_neighbors_and_distance(graph, 5)
    if method == 'HCR':
        return HCR(graph, 0.01)

    if method == 'CAR':
        return CAR(graph)
    if method == 'CRA':
        return CRA(graph)
    if method == 'CAA':
        return CAA(graph)
    if method == 'CJC':
        return CJC(graph)
    if method == 'CPA':
        return CPA(graph)
    if method == 'MI':
        return MI(graph)
    if method == 'MI2':
        return MI2(graph)
    if method == 'MI5':
        return MI5(graph)
    if method == 'CMI':
        return CMI(graph)
    if method == 'CMI2':
        return CMI2(graph)
    if method == 'MA':
        return MA(graph)
    if method == 'MM':
        return MM(graph, 0.001)
    if method == 'MA2':
        return Madm(graph)
    if method == 'MA3':
        return Madm2(graph, 0.001)
    if method == 'LB':
        return (graph)
    else:
        raise Exception('方法错误', method)

###############################################################################

'''
average degree of a network
'''
def average_degree(G):
    s = 0
    for v in G:
        s += nx.degree(G, v)
    # end for

    return s / nx.number_of_nodes(G)
# end def

def pair(x, y):
    if (x < y):
        return (x, y)
    else:
        return (y, x)
    #end if
#end def

'''
insert a ground node into network
'''
def add_ground_node(G):
    node_num = nx.number_of_nodes(G)
    u = node_num
    G.add_node(u)

    for v in range(node_num):
        G.add_edge(u, v)
    # end for
# end def

'''
remove the ground node from network
'''
def remove_ground_node(G):
    node_num = nx.number_of_nodes(G)
    u = node_num - 1

    G.remove_node(u)
# end def

def shortest_path_length(G, source, target, cutoff=None):
    """Compute shortest path length between source and target

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

	target : node label
       ending node for path

    cutoff : integer, optional
        Depth to stop the search.
		If length(source, target) <= cutoff, return length(source, target)
		Else return cutoff + 1

    Returns
    -------
    length : integer
    """
    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    paths = {source: [source]}  # paths dictionary  (paths to key from source)
    if cutoff == 0:
        return 1
    #end if
    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in paths:
                    paths[w] = paths[v] + [w]
                    nextlevel[w] = 1

                    if w == target:
                        return level + 1   # 找的目标节点了
                    #end if
                #end if
            #end for
        #end for
        level = level + 1
        if (cutoff is not None and cutoff <= level):
            break
        #end if
    #end while
    return cutoff + 1
#end def

def dist(u, v, dist_dict):
    if u not in dist_dict.keys():
        return 0
    # end if

    if v not in dist_dict[u].keys():
        return 0
    # end if

    return dist_dict[u][v]
# end def
###############################################################################

"""
Link prediction algorithms.
"""

def trans(m):
    a = [[] for i in m[0]]
    for i in m:
        for j in range(len(i)):
            a[j].append(i[j])
    return a



# Coefficient-variation weight (CV)
def get_weights(matrix, m, n):

    x_mean_list = [sum(matrix[i]) / n for i in range(m)]
    cv_list = [0 for i in range(m)]

    # # 计算平均值
    # for i in range(m):
    #     x = 0
    #     for j in range(n):
    #         x += matrix[i][j]
    #     # end for
    #     x_mean_list[i] = x / n
    # # end for

    # 计算s
    n_minus_one = n - 1
    for i in range(m):
        s = 0
        for j in range(n):
            s += math.pow(matrix[i][j] - x_mean_list[i], 2)
        # end for
        s = s / n_minus_one
        s = math.sqrt(s)

        cv_list[i] = s / x_mean_list[i]
    # end for

    sum_cv = sum(cv_list)
    weight_list = [cv_list[i] / sum_cv for i in range(m)]

    return weight_list

def get_weights2(matrix, m, n):
    weight_list = [0 for i in range(m)]
    alpha = -1 / math.log(n)    # -1/ln(n)
    sum_weight = 0
    for i in range(m):  # 方法的个数
        beta = 0
        for j in range(n):
            r = matrix[i][j]
            try:
                beta += r * math.log(r)
            except ValueError:
                pass
        # end for
        H = alpha * beta
        gamma = 1 - H
        weight_list[i] = gamma
        sum_weight += gamma
    # end for

    for i in range(m):
        weight_list[i] /= sum_weight
    # end for

    return weight_list


def LB(G):
    sim_dict = {}
    node = nx.nodes(G)
    non_edge = nx.non_edges(G)

    block = [i for i in range(nx.number_of_nodes(G))]
    for i in node:
        node_block = []
        node_block.append(i)
        for j in nx.neighbors(G, i):
            node_block.append(j)
        block[i] = node_block

    belong_block = [i for i in range(nx.number_of_nodes(G))]
    for i in node:
        belong_node = []
        for j in block:
            if i in j:
                belong_node.append(j)
        belong_block[i] = belong_node

    for u, v in non_edge:
        s = 0
        block_u = belong_block[u]
        block_v = belong_block[v]
        for i in block_u:
            for j in block_v:
                if (i != j):
                    s += len(set(i) & set(j)) / (len(i) * len(j))
                else:
                    s += 2 / (nx.degree(G, i[0]) + 1)
        sim_dict[(u, v)] = s
    return sim_dict
    
def MA(G):
    sim_dic_ra = resource_allocation_index(G)
    sim_dic_lp = local_path_index(G, 0.001)
    sim_dic_car = CAR(G)
    sim_dic = {}
    ebunch = nx.non_edges(G)
    ra_list = []
    lp_list = []
    car_list = []

    i = 0

    for u, v in ebunch:
        i = i + 1
        ra_list.append(sim_dic_ra[u, v])
        lp_list.append(sim_dic_lp[u, v])
        car_list.append(sim_dic_car[u, v])
    matrix = []
    size = len(lp_list)

    for j in range(size):
        matrix.append([ra_list[j], lp_list[j], car_list[j]])

    matrix = trans(matrix)
    weightlist = get_weights(matrix, 3, size)

    ebunch = nx.non_edges(G)
    for u, v in ebunch:
        sim_dic[u, v] = weightlist[0] * sim_dic_ra[u, v] + weightlist[1] * sim_dic_lp[u, v] + weightlist[2] * \
                        sim_dic_car[u, v]
    return sim_dic

def MM(G, alpha):

    num_basic_method = 3  # 基本方法的个数，RA，CAR，LP

    # 1. 计算LP
    sim_dict = local_path_index(G, alpha)
    size = len(sim_dict)
    sim_matrix = np.zeros((num_basic_method, size))
    index_to_pair_list = [0 for i in range(size)]
    pair_to_index_dict = {}
    square_sum_list = [0 for i in range(num_basic_method)]
    #sum_list = [0 for i in range(num_basic_method)]

    m = 0
    i = 0
    for k in sim_dict.keys():
        s = sim_dict[k]
        pair_to_index_dict[k] = i
        index_to_pair_list[i] = k
        sim_matrix[m][i] = s
        square_sum_list[m] += s * s
        #sum_list[m] += s
        i += 1
    # end for

    m += 1
    # 2. RA
    sim_dict = resource_allocation_index(G)
    for k in sim_dict.keys():
        s = sim_dict[k]
        i = pair_to_index_dict[k]
        sim_matrix[m][i] = s
        square_sum_list[m] += s * s
    # end for

    m += 1
    # 3. CAR
    sim_dict = CAR(G)
    for k in sim_dict.keys():
        s = sim_dict[k]
        i = pair_to_index_dict[k]
        sim_matrix[m][i] = s
        square_sum_list[m] += s * s
       # sum_list[m] += s
    # end for

    sim_dict.clear()

    # normalzie sim_matrix
    for i in range(num_basic_method):
        s = math.sqrt(square_sum_list[i])
        for j in range(size):
            sim_matrix[i][j] /= s
        # end for
    # end for

    # 加权
    weight_list = get_weights(sim_matrix, num_basic_method, size)
    for i in range(num_basic_method):
        sim_matrix[i] *= weight_list[i]
    # end for

    # 求和
    for j in range(size):
        s = 0
        for i in range(num_basic_method):
            s += sim_matrix[i][j]
        # end for

        sim_dict[index_to_pair_list[j]] = s
    # end for

    return sim_dict
# end def
#
def Madm(G):
    sim_dic_ra = resource_allocation_index(G)
    sim_dic_lp = local_path_index(G, 0.001)
    sim_dic_car = CAR(G)
    sim_dic = {}
    ebunch = nx.non_edges(G)
    ra_list = []
    lp_list = []
    car_list = []

    i = 0

    for u, v in ebunch:
        i = i + 1
        ra_list.append(sim_dic_ra[u, v])
        lp_list.append(sim_dic_lp[u, v])
        car_list.append(sim_dic_car[u, v])
    matrix = []
    size = len(lp_list)

    for j in range(size):
        matrix.append([ra_list[j], lp_list[j], car_list[j]])

    t_matrix = trans(matrix)
    weightlist = get_weights(t_matrix, 3, size)

    ebunch = nx.non_edges(G)

    for u, v in ebunch:
        if sim_dic_ra[u, v] == 0:
            sim_dic_ra[u, v] = 0.001
        if sim_dic_lp[u, v] == 0:
            sim_dic_lp[u, v] = 0.001
        if sim_dic_car[u, v] == 0:
            sim_dic_car[u, v] = 0.001
        sim_dic[u, v] = math.exp(weightlist[0] * math.log2(sim_dic_ra[u, v]) + weightlist[1] * math.log2(sim_dic_lp[u, v]) + weightlist[2] * math.log2(sim_dic_car[u, v]))

    return sim_dic

def Madm2(G, alpha):
    num_basic_method = 3  # 基本方法的个数，RA，CAR，LP
    sim_dic = {}
    # 1. 计算LP
    sim_dict_lp = local_path_index(G, alpha)
    size = len(sim_dict_lp)
    sim_matrix = np.zeros((num_basic_method, size))
    index_to_pair_list = [0 for i in range(size)]
    pair_to_index_dict = {}
    square_sum_list = [0 for i in range(num_basic_method)]
    # sum_list = [0 for i in range(num_basic_method)]

    m = 0
    i = 0
    for k in sim_dict_lp.keys():
        s = sim_dict_lp[k]
        pair_to_index_dict[k] = i
        index_to_pair_list[i] = k
        sim_matrix[m][i] = s
        square_sum_list[m] += s * s
        # sum_list[m] += s
        i += 1
    # end for

    m += 1
    # 2. RA
    sim_dict_ra = resource_allocation_index(G)
    for k in sim_dict_ra.keys():
        s = sim_dict_ra[k]
        i = pair_to_index_dict[k]
        sim_matrix[m][i] = s
        square_sum_list[m] += s * s
    # end for

    m += 1
    # 3. CAR
    sim_dict_car = CAR(G)
    for k in sim_dict_car.keys():
        s = sim_dict_car[k]
        i = pair_to_index_dict[k]
        sim_matrix[m][i] = s
        square_sum_list[m] += s * s
    # sum_list[m] += s
    # end for

    #sim_dict_car.clear()

    # normalzie sim_matrix
    for i in range(num_basic_method):
        s = math.sqrt(square_sum_list[i])
        for j in range(size):
            sim_matrix[i][j] /= s
        # end for
    # end for
    
    weight_list = get_weights(sim_matrix, num_basic_method, size)
    
    a =  weight_list[0]
    b =  weight_list[1]
    c =  weight_list[2]

    ebunch = nx.non_edges(G)

    for u, v in ebunch:
        if sim_dict_lp[u, v] == 0:
            sim_dict_lp[u, v] = 0.001
        if sim_dict_ra[u, v] == 0:
            sim_dict_ra[u, v] = 0.001
        if sim_dict_car[u, v] == 0:
            sim_dict_car[u, v] = 0.001
        sim_dic[u, v] = math.exp(a * math.log2(sim_dict_lp[u, v]) + b * math.log2(sim_dict_ra[u, v]) + c * math.log2(sim_dict_car[u, v]))

    return sim_dic
# MI
def MI(G):
     #G = nx.read_edgelist(graph_file)
    
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    nodes = nx.nodes(G)

    beta = -math.log2(0.0001)
    
    # 首先计算$P(L^1_{xy})$，其实不需要计算顶点对之间的概率，只需要不同度之间的概率
    nodes_Degree_dict = {}
    degree_list = []
    for v in nodes:
        nodes_Degree_dict[v] = nx.degree(G, v)
        degree_list.append(nx.degree(G, v))
    
    degree_list = [nx.degree(G, v) for v in nodes]
    distinct_degree_list = list(set(degree_list))
    size = len(distinct_degree_list)
    
    self_Connect_dict = {}
    
    for x in range(size):
        k_x = distinct_degree_list[x]
        for y in range(x, size):
            k_y = distinct_degree_list[y]
            
            p0 = 1
            (k_n, k_m) = pair(k_x, k_y)
            a = edge_num + 1
            b = edge_num - k_m + 1
            for i in range(1, k_n + 1):
                p0 *= (b - i) / (a - i)
            # end for
            if p0 == 1:
                self_Connect_dict[(k_n, k_m)] = beta
                self_Connect_dict[(k_m, k_n)] = beta
            else:
                self_Connect_dict[(k_n, k_m)] = -math.log2(1 - p0)
                self_Connect_dict[(k_m, k_n)] = -math.log2(1 - p0)
            #print (str(k_n) + "," + str(k_m))
            #print (self_Connect_dict[(k_n, k_m)])
# =============================================================================
#     for i in range(size):
#         print(str(i) + "----" + str(distinct_degree_list[i]))
# =============================================================================
    # 计算以z为公共邻居的两个顶点间存在链接的互信息
    #mutual_info_list = [0 for z in range(node_num)]
    
    self_Conditional_dict = {}
    for z in nodes:
        k_z = nodes_Degree_dict[z]
        if k_z > 1:
            alpha = 2 / (k_z * (k_z - 1))
            cc_z = nx.clustering(G, z)
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
                    if i!=j:
                        s += (self_Connect_dict[(nodes_Degree_dict[m], nodes_Degree_dict[n])] - log_c)
            self_Conditional_dict[z] = alpha * s
    
    sim_dict = {}  # 存储相似度的字典
    ebunch = nx.non_edges(G)
    
    for x, y in ebunch:
        s = 0
        #(k_x, k_y) = pair(degree_list[x], degree_list[y])
        for z in nx.common_neighbors(G, x, y):
            s += self_Conditional_dict[z]
        sim_dict[(x, y)] = s - self_Connect_dict[(nodes_Degree_dict[x], nodes_Degree_dict[y])]
        #sim_dict[(y, x)] = s - self_Connect_dict[(nodes_Degree_dict[x], nodes_Degree_dict[y])] 

        # end if
    # end for
    return sim_dict

# end def
def MI2(G):
    #G = nx.read_edgelist(graph_file)
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    sim_dict = {}  # �洢���ƶȵ��ֵ�

    I_pConnect_dict = {}
    pDisConnect = 1
    edges = nx.edges(G)
    ebunch = nx.non_edges(G)
    
    for u, v in edges: 
        uDegree = nx.degree(G, u)
        vDegree = nx.degree(G, v)
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
        mDegree = nx.degree(G, m)
        nDegree = nx.degree(G, n)
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
        sim_dict[(v, u)] = -(I_pConnect - pMutual_Information)
        #print(i)
    return sim_dict

def MI5(G):
    beta = -math.log2(0.0001)
    sim_dict = {}
    edge_num = nx.number_of_edges(G)
    node_num = nx.number_of_nodes(G)
    alpha = -math.log2(edge_num/(node_num*(node_num-1)/2))
    degree_pair = {}
    edge = nx.edges(G)
    nodes = nx.nodes(G)

    nodes_Degree_dict = {}
    degree_list = []
    for v in nodes:
        nodes_Degree_dict[v] = nx.degree(G, v)
        degree_list.append(nx.degree(G, v))

    #计算图中不同的边的个数
    distinct_degree_list = list(set(degree_list))
    size = len(distinct_degree_list)
    for i in range(size):
        for j in range(i, size):
            (di, dj) = pair(distinct_degree_list[i], distinct_degree_list[j])
            degree_pair[di, dj] = 0

    for u, v in edge:
        d1 = nx.degree(G, u)
        d2 = nx.degree(G, v)
        d1, d2 = pair(d1, d2)
        degree_pair[d1,d2] = degree_pair[d1,d2] + 1

    #计算连接的互信息
    self_Connect_dict = {}
    for x in range(size):
        k_x = distinct_degree_list[x]
        for y in range(x, size):
            k_y = distinct_degree_list[y]
            (k_n, k_m) = pair(k_x, k_y)
            if(degree_pair[k_n, k_m] == 0):
                self_Connect_dict[k_n, k_m] = alpha
                self_Connect_dict[k_m, k_n] = alpha
            else:
                self_Connect_dict[k_n, k_m] = -math.log2(degree_pair[k_n, k_m]/edge_num)
                self_Connect_dict[k_m, k_n] = -math.log2(degree_pair[k_n, k_m] / edge_num)

    # 计算以z为公共邻居的两个顶点间存在链接的互信息
    self_Conditional_dict = {}
    for z in nodes:
        k_z = nodes_Degree_dict[z]
        if k_z > 1:
            alpha = 2 / (k_z * (k_z - 1))
            cc_z = nx.clustering(G, z)
            if cc_z == 0:
                log_c = beta
            else:
                log_c = -math.log2(cc_z)
            # end if
            s = 0
            neighbor_list = nx.neighbors(G, z)
            size = len(neighbor_list)
            for i in range(size):
                m = neighbor_list[i]
                for j in range(i + 1, size):
                    n = neighbor_list[j]
                    if i != j:
                        s += (self_Connect_dict[(nodes_Degree_dict[m], nodes_Degree_dict[n])] - log_c)
            self_Conditional_dict[z] = alpha * s

    sim_dict = {}  # 存储相似度的字典
    ebunch = nx.non_edges(G)

    for x, y in ebunch:
        s = 0
        # (k_x, k_y) = pair(degree_list[x], degree_list[y])
        for z in nx.common_neighbors(G, x, y):
            s += self_Conditional_dict[z]
        sim_dict[(x, y)] = s - self_Connect_dict[
            (nodes_Degree_dict[x], nodes_Degree_dict[y])]
    return sim_dict

#考虑公共邻居之间的相连的边
def CMI(G):
    a = 0.1
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    nodes = nx.nodes(G)

    beta = -math.log2(0.0001)

    # 首先计算$P(L^1_{xy})$，其实不需要计算顶点对之间的概率，只需要不同度之间的概率
    nodes_Degree_dict = {}
    degree_list = []
    for v in nodes:
        nodes_Degree_dict[v] = nx.degree(G, v)
        degree_list.append(nx.degree(G, v))

    degree_list = [nx.degree(G, v) for v in nodes]
    distinct_degree_list = list(set(degree_list))
    size = len(distinct_degree_list)

    self_Connect_dict = {}

    for x in range(size):
        k_x = distinct_degree_list[x]
        for y in range(x, size):
            k_y = distinct_degree_list[y]

            p0 = 1
            (k_n, k_m) = pair(k_x, k_y)
            a = edge_num + 1
            b = edge_num - k_m + 1
            for i in range(1, k_n + 1):
                p0 *= (b - i) / (a - i)
            # end for
            if p0 == 1:
                self_Connect_dict[(k_n, k_m)] = beta
                self_Connect_dict[(k_m, k_n)] = beta
            else:
                self_Connect_dict[(k_n, k_m)] = -math.log2(1 - p0)
                self_Connect_dict[(k_m, k_n)] = -math.log2(1 - p0)
            # print (str(k_n) + "," + str(k_m))
            # print (self_Connect_dict[(k_n, k_m)])
    # =============================================================================
    #     for i in range(size):
    #         print(str(i) + "----" + str(distinct_degree_list[i]))
    # =============================================================================
    # 计算以z为公共邻居的两个顶点间存在链接的互信息
    # mutual_info_list = [0 for z in range(node_num)]

    self_Conditional_dict = {}
    for z in nodes:
        k_z = nodes_Degree_dict[z]
        if k_z > 1:
            alpha = 2 / (k_z * (k_z - 1))
            cc_z = nx.clustering(G, z)
            if cc_z == 0:
                log_c = beta
            else:
                log_c = -math.log2(cc_z)
            # end if
            s = 0
            neighbor_list = nx.neighbors(G, z)
            size = len(neighbor_list)
            for i in range(size):
                m = neighbor_list[i]
                for j in range(i + 1, size):
                    n = neighbor_list[j]
                    if i != j:
                        s += (self_Connect_dict[(nodes_Degree_dict[m], nodes_Degree_dict[n])] - log_c)
            self_Conditional_dict[z] = alpha * s

    # 计算节点对公共邻居之间相连的边的个数
    ebunch = nx.non_edges(G)
    neighbor_dict = {}
    for m, n in ebunch:
        com_nei = nx.common_neighbors(G, m, n)
        i = 0
        for x in com_nei:
            for y in com_nei:
                if (m != n) & (G.has_edge(x, y)):
                    i = i + 1
        neighbor_dict[m, n] = i

    sim_dict = {}  # 存储相似度的字典
    ebunch = nx.non_edges(G)

    for x, y in ebunch:
        s = 0
        # (k_x, k_y) = pair(degree_list[x], degree_list[y])
        for z in nx.common_neighbors(G, x, y):
            s += self_Conditional_dict[z]
        sim_dict[(x, y)] = s * (1 + neighbor_dict[x, y]*0.8) - self_Connect_dict[(nodes_Degree_dict[x], nodes_Degree_dict[y])]
    # print(sim_dict)
    return sim_dict

def CMI2(G):
    beta = -math.log2(0.0001)
    sim_dict = {}
    edge_num = nx.number_of_edges(G)
    node_num = nx.number_of_nodes(G)
    alpha = -math.log2(edge_num/(node_num*(node_num-1)/2))
    degree_pair = {}
    edge = nx.edges(G)
    nodes = nx.nodes(G)

    nodes_Degree_dict = {}
    degree_list = []
    for v in nodes:
        nodes_Degree_dict[v] = nx.degree(G, v)
        degree_list.append(nx.degree(G, v))

    #计算图中不同的边的个数
    distinct_degree_list = list(set(degree_list))
    size = len(distinct_degree_list)
    for i in range(size):
        for j in range(i, size):
            (di, dj) = pair(distinct_degree_list[i], distinct_degree_list[j])
            degree_pair[di, dj] = 0

    for u, v in edge:
        d1 = nx.degree(G, u)
        d2 = nx.degree(G, v)
        d1, d2 = pair(d1, d2)
        degree_pair[d1,d2] = degree_pair[d1,d2] + 1

    #计算连接的互信息
    self_Connect_dict = {}
    for x in range(size):
        k_x = distinct_degree_list[x]
        for y in range(x, size):
            k_y = distinct_degree_list[y]
            (k_n, k_m) = pair(k_x, k_y)
            if(degree_pair[k_n, k_m] == 0):
                self_Connect_dict[k_n, k_m] = alpha
                self_Connect_dict[k_m, k_n] = alpha
            else:
                self_Connect_dict[k_n, k_m] = -math.log2(degree_pair[k_n, k_m]/edge_num)
                self_Connect_dict[k_m, k_n] = -math.log2(degree_pair[k_n, k_m] / edge_num)

    # 计算以z为公共邻居的两个顶点间存在链接的互信息
    self_Conditional_dict = {}
    for z in nodes:
        k_z = nodes_Degree_dict[z]
        if k_z > 1:
            alpha = 2 / (k_z * (k_z - 1))
            cc_z = nx.clustering(G, z)
            if cc_z == 0:
                log_c = beta
            else:
                log_c = -math.log2(cc_z)
            # end if
            s = 0
            neighbor_list = nx.neighbors(G, z)
            size = len(neighbor_list)
            for i in range(size):
                m = neighbor_list[i]
                for j in range(i + 1, size):
                    n = neighbor_list[j]
                    if i != j:
                        s += (self_Connect_dict[(nodes_Degree_dict[m], nodes_Degree_dict[n])] - log_c)
            self_Conditional_dict[z] = alpha * s

    # 计算节点对公共邻居之间相连的边的个数
    ebunch = nx.non_edges(G)
    neighbor_dict = {}
    for m, n in ebunch:
        com_nei = nx.common_neighbors(G, m, n)
        i = 0
        for x in com_nei:
            for y in com_nei:
                if (m != n) & (G.has_edge(x, y)):
                    i = i + 1
        neighbor_dict[m, n] = i

    sim_dict = {}  # 存储相似度的字典
    ebunch = nx.non_edges(G)

    for x, y in ebunch:
        s = 0
        # (k_x, k_y) = pair(degree_list[x], degree_list[y])
        for z in nx.common_neighbors(G, x, y):
            s += self_Conditional_dict[z]
        sim_dict[(x, y)] = s * (1 + neighbor_dict[x, y]*0.8) - self_Connect_dict[
            (nodes_Degree_dict[x], nodes_Degree_dict[y])]
    return sim_dict
# CN
def common_neighbors_index(G):
    #print("one time")

    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    # print (node_num)
    # print (edge_num)

    ebunch = nx.non_edges(G)
    
    sim_dict = {}   # 存储相似度的字典
    for u, v in ebunch:
        s = len(list(nx.common_neighbors(G, u, v)))
        # s = 0
        # for w in nx.common_neighbors(G, u, v):
        #     s += 1
        #if (s > 0):
        sim_dict[(u, v)] = s
            #sim_dict[(v, u)] = s
        #end if
    #end for

    return sim_dict
#end def
    

# PA
def preferential_attachment_index(G):
    ebunch = nx.non_edges(G)
    
    sim_dict = {}   # 存储相似度的字典

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for

    for u, v in ebunch:
        s = degree_list[u] * degree_list[v]
        if (s > 0):
            sim_dict[(u, v)] = s
        #end if
    #end for

    return sim_dict
#end def

# Jaccard
def jaccard_coefficient(G):
    ebunch = nx.non_edges(G)
    
    sim_dict = {}   # 存储相似度的字典

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for

    for u, v in ebunch:
        s = len(list(nx.common_neighbors(G, u, v)))
        if (s > 0):
            sim_dict[(u, v)] = s / (degree_list[u] + degree_list[v] - s)
        #end if
    #end for
    
    return sim_dict
#end def

# AA
def adamic_adar_index(G):
    ebunch = nx.non_edges(G)
    
    sim_dict = {}   # 存储相似度的字典

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for
    
    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            s += 1 / math.log2(degree_list[w])
        #end for
        if (s > 0):
            sim_dict[(u, v)] = s
        #end if        
    #end for

    return sim_dict
#end def

# RA
def resource_allocation_index(G):
    ebunch = nx.non_edges(G)
    
    sim_dict = {}   # 存储相似度的字典

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for
    
    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            s += 1 / degree_list[w]
        #end for
        #if (s > 0):
        sim_dict[(u, v)] = s
        #end if        
    #end for

    return sim_dict
#end def

def LNB(G, method):
    """
    @article{Liu2011Link,
      author={Zhen Liu and Qian-Ming Zhang and Linyuan L\"{u} and Tao Zhou},
      title={Link prediction in complex networks: A local naïve Bayes model},
      journal={EPL (Europhysics Letters)},
      volume={96},
      number={4},
      pages={48007},
      url={http://stacks.iop.org/0295-5075/96/i=4/a=48007},
      year={2011}
    }
    """
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    M = node_num * (node_num - 1) / 2
    s = M / edge_num - 1
    logs = math.log2(s)

    degree_list = [nx.degree(G, v) for v in range(G.number_of_nodes())]
    # 计算每个顶点的role
    role_list = [nx.triangles(G, w) for w in range(node_num)]   # 三角形个数
    for w in G:
        triangle = role_list[w]
        numerator = triangle + 1
        d = degree_list[w]
        non_triangle = d * (d - 1) / 2 - triangle
        denominator = non_triangle + 1

        role_list[w] = numerator / denominator
    # end for

    sim_dict = {}  # 存储相似度的字典

    # 计算相似度
    min_value = M
    ebunch = nx.non_edges(G)
    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            if method == 'CN':
                s += logs + math.log2(role_list[w])
            elif method == 'AA':
                s += 1 / math.log2(degree_list[w]) * (logs + math.log2(role_list[w]))
            else:   # RA
                s += 1 / degree_list[w] * (logs + math.log2(role_list[w]))
            # end if
        # end for

        if s != 0:
            sim_dict[(u, v)] = s
            min_value = min(s, min_value)
        # end if
    # end for

    if min_value < 0:
        min_value *= -1
        for k in sim_dict.keys():
            sim_dict[k] += min_value
        # end for
    # end if
    return sim_dict

def CLNB(G, method):
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    alpha = math.log2(1 + edge_num / (node_num * (node_num - 1) / 2))
    degree_pair = {}
    M = node_num * (node_num - 1) / 2
    s = M / edge_num - 1
    logs = math.log2(s)
    edge = nx.edges(G)

    degree_list = [nx.degree(G, v) for v in range(G.number_of_nodes())]
    # 计算每个顶点的role
    role_list = [nx.triangles(G, w) for w in range(node_num)]   # 三角形个数
    for w in G:
        triangle = role_list[w]
        numerator = triangle + 1
        d = degree_list[w]
        non_triangle = d * (d - 1) / 2 - triangle
        denominator = non_triangle + 1

        role_list[w] = numerator / denominator
    # end for

    # 计算图中不同的边的个数
    distinct_degree_list = list(set(degree_list))
    size = len(distinct_degree_list)
    for i in range(size):
        for j in range(i, size):
            (di, dj) = pair(distinct_degree_list[i], distinct_degree_list[j])
            degree_pair[di, dj] = 0

    for u, v in edge:
        d1 = nx.degree(G, u)
        d2 = nx.degree(G, v)
        d1, d2 = pair(d1, d2)
        degree_pair[d1, d2] = degree_pair[d1, d2] + 1

    # 计算连接的互信息
    self_Connect_dict = {}
    for x in range(size):
        k_x = distinct_degree_list[x]
        for y in range(x, size):
            k_y = distinct_degree_list[y]
            (k_n, k_m) = pair(k_x, k_y)
            if (degree_pair[k_n, k_m] == 0):
                self_Connect_dict[k_n, k_m] = alpha
                self_Connect_dict[k_m, k_n] = alpha
            else:
                self_Connect_dict[k_n, k_m] = math.log2(1 + degree_pair[k_n, k_m] / edge_num)
                self_Connect_dict[k_m, k_n] = math.log2(1 + degree_pair[k_n, k_m] / edge_num)

    # 计算节点对公共邻居之间相连的边的个数
    ebunch = nx.non_edges(G)
    neighbor_dict = {}
    for m, n in ebunch:
        com_nei = nx.common_neighbors(G, m, n)
        i = 0
        for x in com_nei:
            for y in com_nei:
                if (m != n) & (G.has_edge(x, y)):
                    i = i + 1
        neighbor_dict[m, n] = i
    sim_dict = {}  # 存储相似度的字典

    # 计算相似度
    min_value = M
    ebunch = nx.non_edges(G)
    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            if method == 'CN':
                s += logs + math.log2(role_list[w])
            elif method == 'AA':
                s += 1 / math.log2(degree_list[w]) * (logs + math.log2(role_list[w]))
            else:   # RA
                s += 1 / degree_list[w] * (logs + math.log2(role_list[w]))
            # end if
        # end for
        if s != 0:
            sim_dict[(u, v)] = s*(1 + neighbor_dict[u, v]*0.05) + self_Connect_dict[degree_list[u], degree_list[v]]
            min_value = min(s, min_value)
        # end if
    # end for

    if min_value < 0:
        min_value *= -1
        for k in sim_dict.keys():
            sim_dict[k] += min_value
        # end for
    # end if
    return sim_dict
    
def CLNB2(G, method):
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    alpha = math.log2(1 + edge_num / (node_num * (node_num - 1) / 2))
    degree_pair = {}
    M = node_num * (node_num - 1) / 2
    s = M / edge_num - 1
    logs = math.log2(s)
    edge = nx.edges(G)
    #degree_list = []
    # for v in range(node_num):
    #     degree_list[v] = nx.degree(G, v)
    degree_list = [nx.degree(G, v) for v in range(node_num)]
    # 计算每个顶点的role
    role_list = [nx.triangles(G, w) for w in range(node_num)]   # 三角形个数
    for w in G:
        triangle = role_list[w]
        numerator = triangle + 1
        d = degree_list[w]
        non_triangle = d * (d - 1) / 2 - triangle
        denominator = non_triangle + 1

        role_list[w] = numerator / denominator
    # end for

    # 计算图中不同的边的个数
    distinct_degree_list = list(set(degree_list))
    size = len(distinct_degree_list)
    for i in range(size):
        for j in range(i, size):
            (di, dj) = pair(distinct_degree_list[i], distinct_degree_list[j])
            degree_pair[di, dj] = 0

    for u, v in edge:
        d1 = nx.degree(G, u)
        d2 = nx.degree(G, v)
        d1, d2 = pair(d1, d2)
        degree_pair[d1, d2] = degree_pair[d1, d2] + 1

    # 计算连接的互信息
    self_Connect_dict = {}
    for x in range(size):
        k_x = distinct_degree_list[x]
        for y in range(x, size):
            k_y = distinct_degree_list[y]
            (k_n, k_m) = pair(k_x, k_y)
            if (degree_pair[k_n, k_m] == 0):
                self_Connect_dict[k_n, k_m] = alpha
                self_Connect_dict[k_m, k_n] = alpha
            else:
                self_Connect_dict[k_n, k_m] = math.log2(1 + degree_pair[k_n, k_m] / edge_num)
                self_Connect_dict[k_m, k_n] = math.log2(1 + degree_pair[k_n, k_m] / edge_num)

    # 计算节点对公共邻居之间相连的边的个数
    ebunch = nx.non_edges(G)
    neighbor_dict = {}
    for m, n in ebunch:
        com_nei = nx.common_neighbors(G, m, n)
        i = 0
        for x in com_nei:
            for y in com_nei:
                if (m != n) & (G.has_edge(x, y)):
                    i = i + 1
        neighbor_dict[m, n] = i
    sim_dict = {}  # 存储相似度的字典

    # 计算相似度
    min_value = M
    ebunch = nx.non_edges(G)
    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            a = 0
            for x in nx.common_neighbors(G, u , v):
                if G.has_edge(w, x):
                    a = a + 1
            if method == 'CN':
                s += (1 + (a/2)) * (logs + math.log2(role_list[w]))
            elif method == 'AA':            
                s += (1 + (a/2))/ math.log2(degree_list[w]) * (logs + math.log2(role_list[w]))
            else:   # RA
                s += (1 + (a/2))/ degree_list[w] * (logs + math.log2(role_list[w]))
            # end if
        # end for
        if s != 0:
            sim_dict[(u, v)] = s #+ self_Connect_dict[degree_list[u], degree_list[v]]
            min_value = min(s, min_value)
        # end if
    # end for

    if min_value < 0:
        min_value *= -1
        for k in sim_dict.keys():
            sim_dict[k] += min_value
        # end for
    # end if
    return sim_dict
'''
@article{cannistraci2013from,
  title={From link-prediction in brain connectomes and protein interactomes to the local-community-paradigm in complex networks},
  author={Cannistraci, Carlo Vittorio and Alanis-Lobato, Gregorio and Ravasi, Timothy},
  journal={Scientific Reports},
  volume={3},
  pages={1613},
  year={2013}
}

$CAR(x, y) = CN(x, y) \cdot \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{|\gamma(z)|}{2}$
'''
# CAR
def CAR(G):
    ebunch = nx.non_edges(G)

    sim_dict = {}  # 存储相似度的字典

    for u, v in ebunch:
        s = 0
        t = 0
        for w in nx.common_neighbors(G, u, v):
            s += 1
            t += len(set(nx.common_neighbors(G, u, w)) & set(nx.common_neighbors(G, v, w)))
        # end for
        #if s > 0:
        t = t / 2.0
        sim_dict[(u, v)] = s * t
        # end if
    # end for

    return sim_dict
# end def

# CRA
def CRA(G):
    ebunch = nx.non_edges(G)

    sim_dict = {}  # 存储相似度的字典

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for

    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            t = len(set(nx.common_neighbors(G, u, w)) & set(nx.common_neighbors(G, v, w)))
            s += t / degree_list[w]
        # end for
        if s > 0:
            sim_dict[(u, v)] = s
        # end if
    # end for

    return sim_dict
# end def

# CAA
def CAA(G):
    ebunch = nx.non_edges(G)

    sim_dict = {}  # 存储相似度的字典

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for

    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            t = len(set(nx.common_neighbors(G, u, w)) & set(nx.common_neighbors(G, v, w)))
            s += t / math.log2(degree_list[w])
        # end for
        if s > 0:
            sim_dict[(u, v)] = s
        # end if
    # end for

    return sim_dict
# end def

# CJC
def CJC(G):
    ebunch = nx.non_edges(G)

    sim_dict = {}  # 存储相似度的字典

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for

    for u, v in ebunch:
        s = 0
        t = 0
        for w in nx.common_neighbors(G, u, v):
            s += 1
            t += len(set(nx.common_neighbors(G, u, w)) & set(nx.common_neighbors(G, v, w)))
        # end for
        if s > 0:
            t = t / 2.0
            sim_dict[(u, v)] = s * t / (degree_list[u] + degree_list[v] - s)
        # end if
    # end for

    return sim_dict
# end def

# CPA
def CPA(G):
    ebunch = nx.non_edges(G)

    sim_dict = {}  # 存储相似度的字典

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for

    for u, v in ebunch:
        s = 0
        t = 0
        for w in nx.common_neighbors(G, u, v):
            s += 1
            t += len(set(nx.common_neighbors(G, u, w)) & set(nx.common_neighbors(G, v, w)))
        # end for
        if s > 0:
            t = t / 2.0
            cc = s * t
            eu = degree_list[u] - s
            ev = degree_list[v] - s
            sim_dict[(u, v)] = eu * ev + (eu + ev) * cc + cc * cc
        # end if
    # end for

    return sim_dict
# end def

'''
@article{Liu2017extended,
title = "Extended resource allocation index for link prediction of complex network",
journal = "Physica A: Statistical Mechanics and its Applications",
volume = "479",
number = "Supplement C",
pages = "174 - 183",
year = "2017",
issn = "0378-4371",
doi = "https://doi.org/10.1016/j.physa.2017.02.078",
url = "http://www.sciencedirect.com/science/article/pii/S0378437117301991",
author = "Shuxin Liu and Xinsheng Ji and Caixia Liu and Yi Bai",
keywords = "Link prediction",
keywords = "Complex network",
keywords = "Resource exchange",
keywords = "Similarity index"
}
'''
# ERA
def extend_resource_allocation_index(G, alpha):
    ebunch = nx.non_edges(G)

    # 计算每个顶点的邻居集
    neighbor_set_list = []
    for x in range(nx.number_of_nodes(G)):
        neighbor_set_list.append(set(nx.neighbors(G, x)))
    #end for

    sim_dict = {}  # 存储相似度的字典
    cn_dict = {}   # 存储两个顶点间公共邻居的个数

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for

    for u, v in ebunch:
        N_u = neighbor_set_list[u]    # u的邻居集
        N_v = neighbor_set_list[v]    # v的邻居集
        CN_uv = N_u & N_v             # u,v的公共邻居的集合
        N_CN_u = N_u - CN_uv          # 与u相连的非公共邻居
        N_CN_v = N_v - CN_uv          # 与v相连的非公共邻居

        if ((u, v) not in cn_dict.keys()):
            cn_dict[(u, v)] = len(CN_uv)
        #end if

        s1 = 0
        s2 = 0
        s3 = 0
        # 计算R(u-->v)+R(v-->u)
        for w in CN_uv:
            (x1, y1) = pair(u, w)
            (x2, y2) = pair(v, w)

            if ((x1, y1) in cn_dict.keys()):
                n1 = cn_dict[(x1, y1)]
            else:
                n1 = len(list(nx.common_neighbors(G, x1, y1)))
                cn_dict[(x1, y1)] = n1
            #end if

            if ((x2, y2) in cn_dict.keys()):
                n2 = cn_dict[(x2, y2)]
            else:
                n2 = len(list(nx.common_neighbors(G, x2, y2)))
                cn_dict[(x2, y2)] = n2
            #end if

            n = n1 + n2
            if (n != 0):
                s1 += (2 + alpha * n) / degree_list[w]
            #end if
        #end for

        # 计算R'(u-->v)
        for w in N_CN_u:
            (x1, y1) = pair(v, w)

            if ((x1, y1) in cn_dict.keys()):
                n1 = cn_dict[(x1, y1)]
            else:
                n1 = len(list(nx.common_neighbors(G, x1, y1)))
                cn_dict[(x1, y1)] = n1
            # end if

            if (n1 != 0):
                s2 += (alpha * n1) / degree_list[w]
            #end if
        # end for

        # 计算R'(v-->u)
        for w in N_CN_v:
            (x1, y1) = pair(u, w)

            if ((x1, y1) in cn_dict.keys()):
                n1 = cn_dict[(x1, y1)]
            else:
                n1 = len(list(nx.common_neighbors(G, x1, y1)))
                cn_dict[(x1, y1)] = n1
            # end if

            if (n1 != 0):
                s3 += (alpha * n1) / degree_list[w]
            # end if
        # end for

        s = s1 + s2 + s3
        #print(s1, s2, s3)
        if (s > 0):
            sim_dict[(u, v)] = s
        # end if
    # end for

    return sim_dict
# end def

'''
 @Article{martinez2016adaptive,
  author   = {Mart\'inez, V\'ictor and Berzal, Fernando and Cubero, Juan-Carlos},
  title    = {Adaptive degree penalization for link prediction},
  journal  = {Journal of Computational Science},
  year     = {2016},
  volume   = {13},
  pages    = {1 - 9},
  issn     = {1877-7503},
  doi      = {http://dx.doi.org/10.1016/j.jocs.2015.12.003},
  keywords = {Link prediction, Networks, Graphs, Topology, Shared neighbors },
  url      = {http://www.sciencedirect.com/science/article/pii/S187775031530051X},
} 
 $s^{ADP}(u, v) = \sum_{w \in \Gamma_u \cap \Gamma_v} \vert \Gamma_w \vert^{-\beta C}$
'''
# ADP
def adaptive_degree_penalization(G, alpha):
    ebunch = nx.non_edges(G)

    C = nx.average_clustering(G)
    param = -1 * alpha * C

    sim_dict = {}   # 存储相似度的字典

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for

    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            s += pow(degree_list[w], param)
        #end for
        if (s > 0):
            sim_dict[(u, v)] = s
        #end if        
    #end for
        
    return sim_dict
#end def

'''
@article{zeng2016link,
title = "Link prediction based on local information considering preferential attachment ",
journal = "Physica A: Statistical Mechanics and its Applications ",
volume = "443",
number = "",
pages = "537 - 542",
year = "2016",
note = "",
issn = "0378-4371",
doi = "http://dx.doi.org/10.1016/j.physa.2015.10.016",
url = "http://www.sciencedirect.com/science/article/pii/S0378437115008626",
author = "Shan Zeng",
keywords = "Link prediction",
keywords = "Complex networks",
keywords = "Similarity index",
keywords = "Node similarity "
}
'''
# CN_PA
def CN_PA(G, alpha):
    ebunch = nx.non_edges(G)

    # 计算每个顶点的度和顶点的平均度
    degree_list = [0 for i in range(G.number_of_nodes())]
    dd = 0
    for v in range(G.number_of_nodes()):
        d = nx.degree(G, v)
        degree_list[v] = d
        dd += d
    #end for

    avg_degree = dd / nx.number_of_nodes(G)
    pp = alpha / avg_degree

    sim_dict = {}  # 存储相似度的字典

    for u, v in ebunch:
        s = len(list(nx.common_neighbors(G, u, v)))
        if (s > 0):
            du = degree_list[u]
            dv = degree_list[v]
            sim_dict[(u, v)] = s + pp * du * dv
        # end if
    # end for

    return sim_dict
# end def

'''
@article{yang2016predicting,
  title={Predicting missing links in complex networks based on common neighbors and distance},
  author={Yang, Jinxuan and Zhang, Xiao-Dong},
  journal={Scientific Reports},
  volume={6},
  pages={38208},
  year={2016},
}
'''
# CNaD
def common_neighbors_and_distance(G, max_length):
    ebunch = nx.non_edges(G)

    sim_dict = {}  # 存储相似度的字典

    dist_dict = nx.all_pairs_shortest_path_length(G, max_length)     # Dictionary of shortest path lengths keyed by source and target

    for u, v in ebunch:
        s = len(list(nx.common_neighbors(G, u, v)))
        # s = 0
        # for w in nx.common_neighbors(G, u, v):
        #     s += 1
        if s > 0:
            sim_dict[(u, v)] = (s + 1) / 2
        else:
            d = dist(u, v, dist_dict)
            #print(u, v, d)
            if d > 0:
                sim_dict[(u, v)] = 1 / d
            # end if
        # end if
    # end for

    return sim_dict
# end def

# LP
def local_path_index(G, alpha):
    ebunch = nx.non_edges(G)

    sim_dict = {}  # 存储相似度的字典
    cn_dict = {}  # 存储两个顶点间公共邻居的个数

    for u, v in ebunch:
        if (u, v) in cn_dict.keys():
            s1 = cn_dict[(u, v)]
        else:
            s1 = len(list(nx.common_neighbors(G, u, v)))
            cn_dict[(u, v)] = s1
        # end if

        # paths with length 3
        s2 = 0
        for w in nx.neighbors(G, u):
            x, y = pair(w, v)
            if (x, y) in cn_dict.keys():
                s2 += cn_dict[(x, y)]
            else:
                ss = len(list(nx.common_neighbors(G, x, y)))
                s2 += ss
                cn_dict[(x, y)] = ss
            # end if
        # end for

        s = s1 + alpha * s2
        #if (s > 0):
        sim_dict[(u, v)] = s
        #end if
    # end for

    return sim_dict
# end def

# HCR
def HCR(G, alpha):
    ebunch = nx.non_edges(G)

    sim_dict = {}  # 存储相似度的字典
    A2 = (nx.to_numpy_matrix(G) ** 2).A  # 转换成矩阵形式

    degree_list = [0 for i in range(G.number_of_nodes())]
    for v in range(G.number_of_nodes()):
        degree_list[v] = nx.degree(G, v)
    # end for

    for u, v in ebunch:
        s1 = 0
        # RA
        for w in nx.common_neighbors(G, u, v):
            s1 += 1 / degree_list[w]
        # end for

        # Correlation
        x = A2[u]
        y = A2[v]
        s2 = stats.pearsonr(x, y)[0]

        s = s1 + s2 * alpha

        if (s > 0):
            sim_dict[(u, v)] = s
            # end if
    # end for

    return sim_dict
# end def

'''
@article{wu2016link,
title = "Link prediction with node clustering coefficient",
journal = "Physica A: Statistical Mechanics and its Applications",
volume = "452",
number = "Supplement C",
pages = "1 - 8",
year = "2016",
issn = "0378-4371",
doi = "https://doi.org/10.1016/j.physa.2016.01.038",
url = "http://www.sciencedirect.com/science/article/pii/S0378437116000777",
author = "Zhihao Wu and Youfang Lin and Jing Wang and Steve Gregory",
keywords = "Link prediction",
keywords = "Complex networks",
keywords = "Clustering coefficient"
}

$CCLP(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} CC_z$
'''
# CCLP
def CCLP(G):
    ebunch = nx.non_edges(G)

    cc_list = [0 for i in range(G.number_of_nodes())]    # 存放每个节点的聚集系数
    for v in range(G.number_of_nodes()):
        cc_list[v] = nx.clustering(G, v)
    # end for

    sim_dict = {}  # 存储相似度的字典

    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            s += cc_list[w]
        # end for
        if s > 0:
            sim_dict[(u, v)] = s
        # end if
    # end for

    return sim_dict
# end def

'''
@article{wu2016predicted,
  author={Zhihao Wu and Youfang Lin and Huaiyu Wan and Waleed Jamil},
  title={Predicting top-L missing links with node and link clustering information in large-scale networks},
  journal={Journal of Statistical Mechanics: Theory and Experiment},
  volume={2016},
  number={8},
  pages={083202},
  url={http://stacks.iop.org/1742-5468/2016/i=8/a=083202},
  year={2016}
}

$\sum_{z \in \Gamma(x) \cap \Gamma(y)} (\frac{CN(x, z) + CN(y, z)}{k_z - 1} \times C_z)$
'''
def NLC(G):
    ebunch = nx.non_edges(G)

    cc_list = [0 for i in range(G.number_of_nodes())]  # 存放每个节点的聚集系数
    for v in range(G.number_of_nodes()):
        cc_list[v] = nx.clustering(G, v)
    # end for

    sim_dict = {}  # 存储相似度的字典

    for u, v in ebunch:
        s = 0
        for w in nx.common_neighbors(G, u, v):
            cn1 = len(list(nx.common_neighbors(G, u, w)))
            cn2 = len(list(nx.common_neighbors(G, v, w)))
            s += (cn1 + cn2) * cc_list[w] / (G.degree(w) - 1)
        # end for
        if s > 0:
            sim_dict[(u, v)] = s
        # end if
    # end for

    return sim_dict
# end def

###############################################################################

def WADP(G, beta = 2.5, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    C = nx.average_clustering(G)
    param = -1 * beta * C
    
    node_num = nx.number_of_nodes(G)
    U = node_num * (node_num - 1) / 2
    
    weight_dict = {}    #存储每条边的权值
    
    for (u, v) in nx.edges(G):
        du = nx.degree(G, u)
        dv = nx.degree(G, v)
        
        weight_dict[(u, v)] = 1 - (du * dv) / U
    #end for
    
    weight_list = []    #存储每个顶点的边的权重之和
    
    for u in range(node_num):
        weight_list.append(0.0)
    #end for
        
    for u in range(node_num):
        weight_list[u] = pow(G.degree(u), param)
    #end for
        
    def predict(u, v):
        s_uv = 0.0
        for w in nx.common_neighbors(G, u, v):
            if (u < w):
                w_uw = weight_dict[(u, w)]
            else:
                w_uw = weight_dict[(w, u)]
            #end if
            if (v < w):
                w_vw = weight_dict[(v, w)]
            else:
                w_vw = weight_dict[(w, v)]
            #end if   
          
            s_uv += (w_uw + w_vw) * weight_list[w]
        #end for
        return s_uv
        
        #return sum(pow(G.degree(w), param) for w in nx.common_neighbors(G, u, v))

    return ((u, v, predict(u, v)) for u, v in ebunch)
#end def
    
"""
Link prediction algorithms.
"""

def WRA(G):
    ebunch = nx.non_edges(G)
    
    node_num = nx.number_of_nodes(G)
    U = node_num * (node_num - 1) / 2
    
    weight_dict = {}    #存储每条边的权值
    
    for (u, v) in nx.edges(G):
        du = nx.degree(G, u)
        dv = nx.degree(G, v)        
        weight_dict[(u, v)] = 1 - (du * dv) / U
    #end for
    
    weight_list = []    #存储每个顶点的边的权重之和
    
    for u in range(node_num):
        weight_list.append(0.0)
    #end for
        
    for u in range(node_num):
        s = 0
        for v in nx.neighbors(G, u):
            if (u > v):
                s += weight_dict[(v, u)]
            else:
                s += weight_dict[(u, v)]
            #end if
        #end for
        weight_list[u] = s
    #end for
        
    sim_dict = {}   # 存储相似度的字典
           
    for u, v in ebunch:     
        s_uv = 0.0
        for w in nx.common_neighbors(G, u, v):
            if (u < w):
                w_uw = weight_dict[(u, w)]
            else:
                w_uw = weight_dict[(w, u)]
            #end if
            if (v < w):
                w_vw = weight_dict[(v, w)]
            else:
                w_vw = weight_dict[(w, v)]
            #end if           
            
            s_uv += (w_uw * w_vw) / weight_list[w]   
        #end for       
        if (s_uv > 0):    
            sim_dict[(u, v)] = s_uv
        #end if
    #end for

    return sim_dict
#end def
    
def WCN(G):
    ebunch = nx.non_edges(G)
    
    node_num = nx.number_of_nodes(G)
    U = node_num * (node_num - 1) / 2
    
    weight_dict = {}    #存储每条边的权值
    
    for (u, v) in nx.edges(G):
        du = nx.degree(G, u)
        dv = nx.degree(G, v)        
        weight_dict[(u, v)] = 1 - (du * dv) / U
    #end for
    
    sim_dict = {}   # 存储相似度的字典
           
    for u, v in ebunch:     
        s_uv = 0.0
        for w in nx.common_neighbors(G, u, v):
            if (u < w):
                w_uw = weight_dict[(u, w)]
            else:
                w_uw = weight_dict[(w, u)]
            #end if
            if (v < w):
                w_vw = weight_dict[(v, w)]
            else:
                w_vw = weight_dict[(w, v)]
            #end if            
          
            s_uv += (w_uw * w_vw)    
        #end for             
        if (s_uv > 0):    
            sim_dict[(u, v)] = s_uv
        #end if
    #end for

    return sim_dict
#end def
    
def LWCN(G):
    ebunch = nx.non_edges(G)
    
    weight_dict = {}    #存储每条边的权值
    
    for (u, v) in nx.edges(G):
        du = nx.degree(G, u)
        dv = nx.degree(G, v)
        cn = len(list(nx.common_neighbors(G, u, v)))
        weight_dict[(u, v)] = (cn * cn) / (du * dv)
    #end for
    
    sim_dict = {}   # 存储相似度的字典
           
    for u, v in ebunch:     
        s_uv = 0.0
        for w in nx.common_neighbors(G, u, v):
            if (u < w):
                lw_uw = weight_dict[(u, w)]
            else:
                lw_uw = weight_dict[(w, u)]
            #end if
            if (v < w):
                lw_vw = weight_dict[(v, w)]
            else:
                lw_vw = weight_dict[(w, v)]
            #end if            
          
            s_uv += (lw_uw + lw_vw)    
        #end for             
        if (s_uv > 0):    
            sim_dict[(u, v)] = s_uv / 2
        #end if
    #end for

    return sim_dict
#end def
    
def WRA2(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

   # M = nx.number_of_edges(G)
    node_num = nx.number_of_nodes(G)
    U = node_num * (node_num - 1) / 2
    
    weight_dict = {}    #存储每条边的权值
    
    for (u, v)  in nx.edges(G):
        du = nx.degree(G, u)
        dv = nx.degree(G, v)
        
        weight_dict[(u, v)] = 1 - (du * dv) / U
    #end for
    
    weight_list = []    #存储每个顶点的边的权重之和
    
    for u in range(node_num):
        weight_list.append(0.0)
    #end for
        
    for u in range(node_num):
        s = 0
        for v in nx.neighbors(G, u):
            if (u > v):
                s += weight_dict[(v, u)]
            else:
                s += weight_dict[(u, v)]
            #end if
        #end for
        weight_list[u] = s
        
    def predict(u, v):
        s_uv = 0.0
        for w in nx.common_neighbors(G, u, v):
            if (u < w):
                w_uw = weight_dict[(u, w)]
            else:
                w_uw = weight_dict[(w, u)]
            #end if
            if (v < w):
                w_vw = weight_dict[(v, w)]
            else:
                w_vw = weight_dict[(w, v)]
            #end if           
            s_w = weight_list[w]
            
            s_uv += (w_uw + w_vw) / s_w            
        #end for
        return s_uv

    return ((u, v, predict(u, v)) for u, v in ebunch)
#end def
    
def WCN2(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

   # M = nx.number_of_edges(G)
    node_num = nx.number_of_nodes(G)
    U = node_num * (node_num - 1) / 2
    
    weight_dict = {}    #存储每条边的权值
    
    for (u, v)  in nx.edges(G):
        du = nx.degree(G, u)
        dv = nx.degree(G, v)
        
        weight_dict[(u, v)] = 1 - (du * dv) / U
    #end for
              
    def predict(u, v):
        s_uv = 0.0
        for w in nx.common_neighbors(G, u, v):
            if (u < w):
                w_uw = weight_dict[(u, w)]
            else:
                w_uw = weight_dict[(w, u)]
            #end if
            if (v < w):
                w_vw = weight_dict[(v, w)]
            else:
                w_vw = weight_dict[(w, v)]
            #end if           
                        
            s_uv += (w_uw + w_vw)
        #end for
        return s_uv

    return ((u, v, predict(u, v)) for u, v in ebunch)
#end def
    
###############################################################################


'''
common neighbores index with ground node
'''        
def GCN(G):
    
    add_ground_node(G)
    
    return common_neighbors_index(G)
#end def
    
'''
adamic_adar_index with ground node
'''        
def GAA(G):
    
    add_ground_node(G)
    
    return adamic_adar_index(G)
#end def
    
'''
resource_allocation_index with ground node
'''        
def GRA(G):
    
    add_ground_node(G)
    
    return resource_allocation_index(G)
#end def
    
'''
common neighbores index with ground node
'''        
def GCNPA(G):
    
#    print(nx.number_of_nodes(G))
#    print(nx.number_of_edges(G))
    add_ground_node(G)
    
#    print(nx.number_of_nodes(G))
#    print(nx.number_of_edges(G))
    
    return CNPA(G)   
#end def
    
'''
adamic_adar_index with ground node
'''        
def GAAPA(G):
    
    add_ground_node(G)
    
    return AAPA(G)   
#end def
    
'''
resource_allocation_index with ground node
'''        
def GRAPA(G):
    
    add_ground_node(G)
    
    return RAPA(G)
#end def
    
def common_neighbors(G, u, v):
    
    s = 0
    for w in nx.common_neighbors(G, u, v):
        s += 1
    #end for
    return s
#end def

def resource_allocation(G, u, v):
    
    s = 0
    for w in nx.common_neighbors(G, u, v):
        s += 1 / G.degree(w)
    #end for
    return s
#end def
    
def ERA(G):
    
    ebunch = nx.non_edges(G)
    
    sim_dict = {}   # 存储相似度的字典
    
    for u, v in ebunch:
        sim_dict[(u, v)] = resource_allocation(G, u, v)
        
        # paths with length 3
        s = 0
        for w in nx.neighbors(G, u):
            s += (1 / G.degree(w)) * resource_allocation(G, w, v)
        #end for
        sim_dict[(u,v)] += s
    #end for        

    return sim_dict
#end def