def MI(G):
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)

    beta = math.log2(0.0001)

    degree_list = [nx.degree(G, v) for v in range(G.number_of_nodes())]

    # 首先计算$P(L^1_{xy})$，其实不需要计算顶点对之间的概率，只需要不同度之间的概率
    self_info_dict = {}
    distinct_degree_list = list(set(degree_list))
    size = len(distinct_degree_list)
    for x in range(1, size + 1):
        k_x = distinct_degree_list[x]
        for y in range(x, size + 1):
            k_y = distinct_degree_list[y]

            p0 = 1
            (k_n, k_m) = pair(k_x, k_y)
            a = edge_num + 1
            b = edge_num - k_m + 1
            for i in range(1, k_n + 1):
                p0 *= (b - i) / (a - i)
            # end for
            if p0 == 1:
                self_info_dict[(k_n, k_m)] = -beta
            else:
                self_info_dict[(k_n, k_m)] = -math.log2(1 - p0)
            # end if
        # end for
    # end for

    # 计算以z为公共邻居的两个顶点间存在链接的互信息
    mutual_info_list = [0 for z in range(node_num)]
    for z in G:
        k_z = degree_list(z)
        alpha = 1 / (k_z * (k_z - 1))
        cc_z = nx.clustering(G, z)
        if cc_z == 0:
            log_c = beta
        else:
            log_c = math.log2(cc_z)
        # end if

        neighbors = nx.neighbors(G, z)
        size = len(neighbors)
        s = 0
        for i in range(0, size - 1):
            m = neighbors[i]
            for j in range(i + 1, size):
                n = neighbors[j]
                (k_n, k_m) = pair(degree_list[n], degree_list[m])
                s += self_info_dict[(k_n, k_m)] + log_c
            # end for
        # end for
        mutual_info_list[z] = alpha * s
    # end for

    sim_dict = {}  # 存储相似度的字典
    ebunch = nx.non_edges(G)

    for x, y in ebunch:
        s = 0
        (k_x, k_y) = pair(degree_list[x], degree_list[y])
        for z in nx.common_neighbors(G, x, y):
            s += mutual_info_list[z] - self_info_dict[(k_x, k_y)]
        if s > 0:
            sim_dict[(x, y)] = s
        # end if
    # end for

    return sim_dict
# end def
