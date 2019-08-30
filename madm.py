import networkx as nx
import math
from MA import get_weights
import sim2
import numpy as np

def trans(m):
    a = [[] for i in m[0]]
    for i in m:
        for j in range(len(i)):
            a[j].append(i[j])
    return a

def madm(G):
    sim_dic_ra = sim2.resource_allocation_index(G)
    sim_dic_lp = sim2.local_path_index(G, 0.001)
    sim_dic_car = sim2.CAR(G)
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

def madm2(G, alpha):
    num_basic_method = 3  # 基本方法的个数，RA，CAR，LP

    # 1. 计算LP
    sim_dict_lp = sim2.local_path_index(G, alpha)
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
    sim_dict_ra = sim2.resource_allocation_index(G)
    for k in sim_dict_ra.keys():
        s = sim_dict_ra[k]
        i = pair_to_index_dict[k]
        sim_matrix[m][i] = s
        square_sum_list[m] += s * s
    # end for

    m += 1
    # 3. CAR
    sim_dict_car = sim2.CAR(G)
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
    sim_dic = []
    weight_list = get_weights(sim_matrix, num_basic_method, size)
    ebunch  = nx.non_edges(G)
    for u, v in ebunch:
        sim_dic[u,v] = weight_list[0]*sim_dict_lp[u,v] + weight_list[1]*sim_dict_ra[u, v] + weight_list[2]*sim_dict_car[u,v]

G = nx.Graph()
G.add_edges_from([(1,4),(1,3),(1,2),(2,5),(2,4),(5,7),(5,6),(6,7),(6,8),(7,8)])
madm(G)