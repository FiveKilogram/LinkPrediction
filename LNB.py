'''
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
'''
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