def MI(G):
    #G = nx.read_edgelist(graph_file)
   # G = nx.read_edgelist(graph_file, nodetype=int)# 将点类型改为int，使点标示和序号对应
    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    
    nodes = nx.nodes(G)

    beta = -math.log2(0.0001)
    
    # 首先计算$P(L^1_{xy})$，其实不需要计算顶点对之间的概率，只需要不同度之间的概率
    #nodes_Degree_dict = {}
    degree_list = []
    
# =============================================================================
#     for v in nodes:
#         nodes_Degree_dict[v] = nx.degree(G, v)
#         degree_list.append(nx.degree(G, v))
# =============================================================================
    
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
    
    # 计算以z为公共邻居的两个顶点间存在链接的互信息
    #mutual_info_list = [0 for z in range(node_num)]
    
    self_Conditional_dict = {}
    for z in nodes:
        k_z = degree_list[z]
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
                        s += (self_Connect_dict[(degree_list[m], degree_list[n])] - log_c)
            self_Conditional_dict[z] = alpha * s
    
    sim_dict = {}  # 存储相似度的字典
    ebunch = nx.non_edges(G)
    
    i = 0
    for x, y in ebunch:
        s = 0
        #(k_x, k_y) = pair(degree_list[x], degree_list[y])
        for z in nx.common_neighbors(G, x, y):
            s += self_Conditional_dict[z]
        sim_dict[(x, y)] = s - self_Connect_dict[(degree_list[x], degree_list[y])]
        #sim_dict[(y, x)] = s - self_Connect_dict[(degree_list[x], degree_list[y])]
        # end if
    # end for
    return sim_dict