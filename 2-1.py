# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 19:28:15 2018

@author: Administrator
"""

import networkx as nx
import random
import numpy as np
import math

#读取groupmemberships，节点、标签
def readfile(filename):
    sim_linelist={}
    sim_labeldict={}
    labellist = []
    b_number = '1'#第一列节点
    with open(filename,'r') as f:  
        for line in f.readlines():
#            print(line)
            linestr = line.strip()  
#            print(linestr)  
            linestrlist = linestr.split("\t")  
#            print(linestrlist) #输出['',''] 
            linelist = map(int,linestrlist)# linelist = [int(i) for i in linestrlist]  
#            print(linelist) 
            sim_linelist[line]=linelist
            a=np.array(linestrlist)
            gnode_array=a[0]
            glabel_array=a[1]
#            print(a[0])
#            print(a[1])
            gnode_list=gnode_array.tolist()
            glabel_list=glabel_array.tolist()
            
            if(gnode_list != b_number):
                labellist = []#清空列表
                labellist.append(glabel_list)#输出{‘’：['']} 节点：标签集
            else:
                labellist.append(glabel_list)
            
            sim_labeldict[gnode_list] = labellist#节点+标签字典
            b_number = str(gnode_list)
#            print(gnode_list)
#            print(glabel_list)
    return sim_labeldict
sim_labeldict = readfile('release-youtube-groupmemberships.txt')
#print(sim_labeldict)


#读入网络links，随机选择节点
G=nx.Graph(nx.read_edgelist('release-youtube-links.txt'))
ledge_list=nx.edges(G)#edges=nx.edges(G) #ledge_list=edges
lnode_list=nx.nodes(G)#nodes=nx.nodes(G) #lnode_list=nodes
#print(ledge_list)
#print(lnode_list)
randomnode = random.choice(lnode_list)#随机节点
#print(randomnode)
randomnode_neighbor = {randomnode}#随机节点邻居
#print(randomnode_neighbor)
while(len(randomnode_neighbor) < 50):
    for v in randomnode_neighbor:
        randomnode_neighbor = randomnode_neighbor | set(G.neighbors(v))
        if len(randomnode_neighbor) > 50:
            break
#print(randomnode_neighbor)
#print(len(randomnode_neighbor))

#生成随机节点图
g=nx.Graph()
for u in randomnode_neighbor:
    for v in randomnode_neighbor:
        if(G.has_edge(u, v)):
            g.add_edge(u, v)
g_randomedges=nx.edges(g)
#print(g_randomedges)
#==============================================================================
# node_num=nx.number_of_nodes(g)
# #print(node_num)
# edge_num=nx.number_of_edges(g)
# M = node_num * (node_num - 1) / 2
# s = M / edge_num - 1
# logs = math.log2(s)
#==============================================================================



#将groupmemberships的标签加入到links随机节点图中

key_node=list(sim_labeldict.keys())#sim_labeldict中的节点  gnode_list
#print(key_node)
#slabel_list=list(sim_labeldict.values())#sim_labeldict中的标签  glabel_list
#print(slabel_list)
lnode_dict={}#随机节点加上标签
for v in randomnode_neighbor:
#    print (v)    
    if v in key_node:
#        lnode_dict[v]=sim_labeldict.get(v)
        lnode_dict[v]=sim_labeldict[v]
#         print(lnode_dict)
    else:        
        lnode_dict[v]=[] #sim_labeldict.get(0)#lnode_dict[v]=['']
#print(lnode_dict)

def cluster(g,z):
    for z in g:
        lnode3=lnode_dict[z]
        neighbors=nx.neighbors(g,z)#z的邻居                
#                print(neighbors)
        degree = 0
        neighbor_list=[]
        for w in neighbors:#z的邻居
            lnode4=lnode_dict[w]
            if len(set(lnode3) & set(lnode4))>0:#与z有共同标签
                degree = degree+1#度
                neighbor_list.append(w)
#                print(degree) 
        triangle=0
        size = len(neighbor_list)
        for u in range(size):
            lnode1=lnode_dict[neighbor_list[u]]
            for v in range(u+1,size):
                lnode2=lnode_dict[neighbor_list[v]]
                if (g.has_edge(u,v))&(len(set(lnode1) & set(lnode2))>0):
                    triangle=triangle+1#三角形
                    
        N = (triangle+1)/((degree*(degree-1)/2)-triangle+1)
#        print(N)
    return N,degree
#N,degree=cluster(g,z)   
#print(N)

def GNB(g,method):
    node_num=nx.number_of_nodes(g)    
    edge_num=nx.number_of_edges(g)
    M = node_num * (node_num - 1) / 2
    s = M / edge_num - 1
    logs = math.log2(s)
    
    
    sim_dict = {}  # 存储相似度的字典

    # degree_list = []
#    if method != 'CN':
#        degree_list = degree_of_nodes()#???
    # end if

    # 计算相似度
    ebunch = nx.non_edges(g)
    for u, v in ebunch:
        s = 0
        lnode1=lnode_dict[u]#u的标签
        lnode2=lnode_dict[v]#v的标签            
        if len(set(lnode1) & set(lnode2))>0:
            for z in nx.common_neighbors(g, u, v):
                N, degree =  cluster(g,z)
                if method == 'CN':
                    s += logs + math.log2(N)
                elif method == 'AA':
                    s += 1 / math.log2(degree) * (logs + math.log2(N))
                else:   # RA
                    s += 1 / degree * (logs + math.log2(N))
            if s > 0:
                sim_dict[(u, v)] = s
    print(sim_dict)
    return sim_dict
#sim_dict=GNB(g,method)

GNB(g,'CN')












