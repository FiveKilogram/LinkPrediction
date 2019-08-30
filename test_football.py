# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:47:22 2018

@author: Administrator
"""

import networkx as nx
import random
import numpy as np
#==============================================================================
# G=nx.read_edgelist('dataset\\football_12_links.txt')
# edges=nx.edges(G)
# nodes=nx.nodes(G)
# randomnode = random.choice(nodes)#随机节点
# print(randomnode)
# randomnode_neighbor = {randomnode}#随机节点邻居
# while(len(randomnode_neighbor) < 50):
#     for v in randomnode_neighbor:
#         randomnode_neighbor = randomnode_neighbor | set(G.neighbors(v))
#         if len(randomnode_neighbor) > 50:
#             break
# print(randomnode_neighbor)#随机节点邻居
# 
# sim_linelist={}
# node_list=[]
# label_list=[]
# def readfile(filename):  
#     with open(filename,'r') as f:  
#         for line in f.readlines():
# #            print(line)
#             linestr = line.strip()  
#             print(linestr)  
#             linestrlist = linestr.split("\t")
# #            print(linestrlist)  
#             node_list.append(linestrlist[0])
# #            label_list.append(linestrlist[1])
#             print(node_list)
# #            print(label_list)
# 
#             linelist = map(int,linestrlist)# 方法一  
#             # linelist = [int(i) for i in linestrlist] # 方法二  
# #            print(linelist) 
#             sim_linelist[line]=linelist
# #            print(sim_linelist)
# #vs = G.node
# #neigdic = {}
# #for v in neig:
# #    neigdic[v] = vs[v]['label']
# #print(neigdic)
# 
# readfile('dataset\\football_12_groupmemberships.txt')
#==============================================================================

#==============================================================================
# node_list = []
# label_list = []  
# def loadData(flieName):  
#     with open(flieName, 'r') as inFile:#以只读方式打开某fileName文件  
#     #定义两个空list，用来存放文件中的数据        
#         for line in inFile:  
#             trainingSet = line.split() #对于每一行，把数据分开  
#             node_list.append(trainingSet[0]) #第一列数据逐一添加到list X 中  
#             label_list.append(trainingSet[1]) #第二列数据逐一添加到list y 中  
# #        return (node_list,label_list)# X,y组成一个元组，通过函数一次性返回 
#             print(node_list)        
#             print(label_list)
# loadData('dataset\\football_12_groupmemberships.txt')
#==============================================================================

#读取groupmemberships，节点、标签




#labellist = []
def readfile(filename):
    sim_linelist={}
    sim_labeldict={}
    labellist = []
    b_number = '1'
    with open(filename,'r') as f:  
        for line in f.readlines():
#            print(line)
            linestr = line.strip()  
#            print(linestr)  
            linestrlist = linestr.split("\t")  
#            print(linestrlist) #输出['',''] 
            linelist = map(int,linestrlist)# 方法一  
            # linelist = [int(i) for i in linestrlist] # 方法二  
#            print(linelist) 
            sim_linelist[line]=linelist
            a=np.array(linestrlist)
            gnode_array=a[0]
            glabel_array=a[1]
#            print(a[0])
#            print(a[1])
#            print(a[0],a[1])
            gnode_list=gnode_array.tolist()
            glabel_list=glabel_array.tolist()
            
            if(gnode_list != b_number):
                labellist = []
                labellist.append(glabel_list)
            else:
                labellist.append(glabel_list)
            
            sim_labeldict[gnode_list] = labellist
            b_number = str(gnode_list)
#            print(gnode_list)
#            print(glabel_list)
#            linestrlist1=[linestrlist]
#            print(linestrlist1)
#            sim_labeldict=linestrlist
#            print(sim_linestrdict)
    print(sim_labeldict)
readfile('release-youtube-groupmemberships.txt')

#读入网络links，随机选择节点
G=nx.read_edgelist('dataset\\release-youtube-links.txt')
edges=nx.edges(G)
nodes=nx.nodes(G)
#print(edges)
#print(nodes)
ledge_list=[edges]
lnode_list=[nodes] 
print(ledge_list)
print(lnode_list)

for v in lnode_list:
    if v in gnode_list:
        
    
        node_list=lnode_list.append(glabel_list[v])
        print(node_list)
    








