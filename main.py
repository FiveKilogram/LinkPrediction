# -*- coding: utf-8 -*-
"""
Personalized Link Prediction

Created on Fri Oct 21 09:33:16 2016

@author: Longjie Li
"""


# import math
import lp
#import test


'''----------------------------------------------------------------------------
参数设置
'''
t = 50		# 独立实验的次数
p = 0.1		# 测试数据的比例，测试集的大小
# alpha = 0.01	# 部分算法中的参数
# beta = 0.01		# 部分算法中的参数
suf = str(p * 10) +'-mm'
path = '.\\Networks\\'  # 网络数据的根目录

# 网络文件
networks = [
    'CE\\celegansneural.edgelist', 		# 0
	'jazz\\jazz.edgelist',				# 1
    'karate\\karate.edgelist',  		# 2
    'netscience\\netscience.edgelist',  # 3
    'USAir\\USAir97.edgelist',  		# 4
    'power\\power.edgelist',  			 # 5  12197676
    'yeast\\yeast.edgelist',  			 # 6  2465367
    'PGP\\PGP.edgelist',  				    # 7  57001544
    'Dolphins\\dolphins.edgelist',  	 # 8
    'PB\\polblogs.edgelist',  			 # 9
    'email\\email.edgelist',  			 # 10
    'Hamster\\Hamster.edgelist',  		 # 11 1585102
    'HEP\\hep.edgelist',                # 12 17006880
    'word\\word.edgelist',              # 13
    'foodweb\\baywet.edgelist',         # 14
    'Router\\Router.edgelist',          # 15 12601473
    'INF\\INF.edgelist',                # 16
    'FBK\\FaceBook.edgelist',           # 17 7970223
    'ADV\\ADV.edgelist',                # 18 12669134
    'Wikivote\\Wikivote.edgelist',      # 19 25207293
    'SciMet\\SciMet.edgelist',          # 20
    'Florida\\Florida.edgelist',      # 21
    'Kohonen\\Kohonen.net',      # 22
    'Mangdry\\mangdry.edgelist',      # 23
    'Mangwet\\mangwet.edgelist',      # 24
    'Lederberg\\Lederberg.edgelist',      # 25
    'GrQc\\GrQc.edgelist',              # 26
    'Metabolic\\metabolic.edgelist',    # 27
    'OpenFlights\\openflights.edgelist',# 28
    'SmallGri\\SmallGri.edgelist',      # 29
    'test\\test.edgelist'               #
]

# 对应的结果文件

results = [
    '.\\results\\CE-',
    '.\\results\\Jazz-',
    '.\\results\\Karate-',
    '.\\results\\NS-',
    '.\\results\\USAir-',
    '.\\results\\Power-',
    '.\\results\\Yeast-',
    '.\\results\\PGP-',
    '.\\results\\Dolphins-',
    '.\\results\\PB-',
    '.\\results\\Email-',
    '.\\results\\Hamster-',
    '.\\results\\HEP-',
    '.\\results\\Word-',
    '.\\results\\FoodWeb-',
    '.\\results\\Router-',
    '.\\results\\INF-',
    '.\\results\\FaceBook-',
    '.\\results\\ADV-',
    '.\\results\\Wikivote',
    '.\\results\\SciMet',
    '.\\results\\Florida',
    '.\\results\\Kohonen',
    '.\\results\\Mangdry',
    '.\\results\\Mangwet',
    '.\\results\\Lederberg',
    '.\\results\\GrQc-',
    '.\\results\\Metabolic-',
    '.\\results\\Flight-',
    '.\\results\\SmallGri-',
    '.\\results\\test-',
]

# 实验中可能只使用部分网络， 下面数组中指定相应网络的id
net_ids = [2]#,  7, 8, 15,  17, 18]#
#net_ids = [26,27,28,29]
#net_ids = [6, 11]
graph_file_list = []  # 网络文件列表
result_file_list = []  # 结果文件列表

for i in net_ids:
    graph_file_list.append(path + networks[i])
    result_file_list.append(results[i] + suf)
# end for

# 相似性方法
sim_methods = [
    'CN',  		# 0
    'AA',  		# 1
    'RA',  		# 2
    'Jaccard',  # 3
    'PA',  		# 4
    'CN_PA',  	# 5
    'ADP',  	# 6
    'CNaD',  	# 7
    'ERA',  	# 8
    'LP',  		# 9
    'HCR',      # 10

    'CAR',		# 11
    'CRA',      # 12
    'CAA',      # 13
    'CPA',      # 14
    'CJC',      # 15

    'LCCL',     # 16
    'CCLP',     # 17
    'NLC',      # 18

    'LNB_CN',   # 19
    'LNB_AA',   # 20
    'LNB_RA',   # 21
    'MI',       # 22

    'PNR_CN',   # 23
    'PNR_AA',   # 24
    'PNR_RA',   # 25
    'PNR_JC',   # 26

    'MAX',       # 27
    'TOP',       # 28
    'ED_CN',    # 29
    'ED_AA',    # 30
    'ED_RA',    # 31
    'MI',    # 32
    'CMI',    # 33
    'CMI2',    # 34
    'MM',       # 35
    'MA',       # 36
    'MA2',      # 37
    'MI5',     # 38
    'CLNB_CN',     # 39
    'CLNB2_CN',     # 40
    'CLNB2_AA',     # 41
    'CLNB2_RA',     # 42
    'LPP',#43
    'L',#44
]

# 实验中使用的方法的id
method_ids = [0]
sim_method_list = [sim_methods[i] for i in method_ids]

# 按照数据集，分别计算
for i in range(len(graph_file_list)):
    graph_file = graph_file_list[i]
    result_file = result_file_list[i]
    out_file = open(result_file, 'w')  # 打开结果文件

    print(graph_file)
    # 输出标题
    out_file.write('Method\tAUC\tRanking_Score\ttime (ms)\tPrecision (10)\n')

    # 按照不同的相似度方法分别计算
    for method in sim_method_list:
        print(method)
        out_file.write(method + '\t')
        lp.LP(graph_file, out_file, method, t, p)
        out_file.flush()
    # end for
    out_file.close()
# end for

###############################################################################
