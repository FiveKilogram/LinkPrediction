import networkx as nx
import math
import sim2

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
    

# entropy method
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
    
# =============================================================================
# a = [[0.72,0.69,0.71],[0.72,0.69,0.73],[0.72,0.69,0.72]]
# wlist = get_weights(a,3,3)
# print(wlist)
# =============================================================================
def MA(G):  
    sim_dic_cn = sim2.common_neighbors_index(G)
    sim_dic_lp = sim2.local_path_index(G,0.001)
    sim_dic_car = sim2.CAR(G)
    sim_dic = {}
    ebunch = nx.non_edges(G)
    cn_list = []
    lp_list = []
    car_list = []

    i = 0

    for u, v in ebunch:
        i = i + 1
        cn_list.append(sim_dic_cn[u,v])
        lp_list.append(sim_dic_lp[u,v])
        car_list.append(sim_dic_car[u,v])
    matrix = []
    size = len(lp_list)
    
    for j in range(size):
        matrix.append([cn_list[j],lp_list[j],car_list[j]])
        
    matrix = trans(matrix)
    weightlist = get_weights(matrix,3,size)

    print(weightlist)
    ebunch = nx.non_edges(G)
    for u, v in ebunch: 
        sim_dic[u,v] = weightlist[0]*sim_dic_cn[u,v] + weightlist[1]*sim_dic_lp[u, v] + weightlist[2]*sim_dic_car[u,v]

    return sim_dic



G = nx.Graph()
G.add_edges_from([(1,4),(1,3),(1,2),(2,5),(2,4),(5,7),(5,6),(6,7),(6,8),(7,8)])
MA(G)

# G = nx.read_edgelist('.\\Networks\\CE\\celegansneural.edgelist')
# MA(G)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




