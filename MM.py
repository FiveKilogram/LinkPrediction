import math
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