import math
import networkx as nx

def pair(x, y):
    if (x < y):
        return (x, y)
    else:
        return (y, x)
#改变两个节点之间连接概率的计算方法
def CLNB2(G, method):
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    alpha = math.log2(edge_num / (node_num * (node_num - 1) / 2))
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
                self_Connect_dict[k_n, k_m] = math.log2(degree_pair[k_n, k_m] / edge_num)
                self_Connect_dict[k_m, k_n] = math.log2(degree_pair[k_n, k_m] / edge_num)

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
            sim_dict[(u, v)] = s*(1 + neighbor_dict[u, v]) + self_Connect_dict[degree_list[u], degree_list[v]]
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