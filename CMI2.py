import networkx as nx
import math
def pair(x, y):
    if (x < y):
        return (x, y)
    else:
        return (y, x)

def CMI2(G):
    beta = 0.001
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
        sim_dict[(x, y)] = s * (1 + neighbor_dict[x, y]*1) - self_Connect_dict[
            (nodes_Degree_dict[x], nodes_Degree_dict[y])]
    print(sim_dict)
    return sim_dict

#G = nx.Graph()
#G.add_edges_from([(1,4),(1,3),(1,2),(2,5),(2,4),(5,7),(5,6),(6,7),(6,8),(7,8)])
#CMI2(G)
#CMI2(nx.read_edgelist('J:\\Python\\LinkPrediction\\Networks\\test\\test.edgelist'))
G = nx.read_edgelist('F:\\LinkPrediction\\Networks\\netscience\\netscience.edgelist')
CMI2(G)