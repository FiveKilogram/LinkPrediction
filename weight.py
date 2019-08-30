import math

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
# end def

# entropy method
def get_weights(matrix, m, n):
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
# end for