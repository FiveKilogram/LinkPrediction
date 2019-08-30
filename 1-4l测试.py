# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:22:37 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:47:22 2018

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
while(len(randomnode_neighbor) < 50000):
    for v in randomnode_neighbor:
        randomnode_neighbor = randomnode_neighbor | set(G.neighbors(v))
        if len(randomnode_neighbor) > 50000:
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
node_num=nx.number_of_nodes(g)
#print(node_num)
edge_num=nx.number_of_edges(g)
M = node_num * (node_num - 1) / 2
s = M / edge_num - 1
logs = math.log2(s)



#将groupmemberships的标签加入到links随机节点图中

key_node=list(sim_labeldict.keys())#sim_labeldict中的节点  gnode_list
#print(key_node)
#slabel_list=list(sim_labeldict.values())#sim_labeldict中的标签  glabel_list
#print(slabel_list)
lnode_dict={}#随机节点加上标签
for v in randomnode_neighbor:
#    print (v)    
    if v in key_node:
        #lnode_dict[v]=sim_labeldict.get(v)
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
        if degree>0:
            print(str(z)+":"+str(degree))
                #print(degree) 
        triangle=0
        size = len(neighbor_list)
        for u in range(size):
            lnode1=lnode_dict[neighbor_list[u]]
            for v in range(u+1,size):
                lnode2=lnode_dict[neighbor_list[v]]
                if len(set(lnode1) & set(lnode2))>0:
                    triangle=triangle+1#三角形
        if degree>0:
            print(str(z)+":"+str(triangle))
                    
        N = (triangle+1)/((degree*(degree-1)/2)-triangle+1)
        #print(N)
    return N,degree

node = nx.nodes(g)
for m in node:
    n, degree = cluster(g,m)
    #print(str(m)+":"+str(n)+"--"+str(degree))
# =============================================================================
# N,degree=cluster(g,z)   
# #print(N)
# 
# def GNB(g,method):
#     node_num=nx.number_of_nodes(g)
#     print(node_num)
# =============================================================================




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
# #找u,v是邻居下的共同标签
# for u in lnode_dict:
#     lnode1=lnode_dict[u]
#     for v in lnode_dict:
#         lnode2=lnode_dict[v]
#         if (g.has_edge(u,v)):
#             print(u,v)
#             print(lnode_dict[u],lnode_dict[v])
#             lnode=set(lnode1) & set(lnode2)
#             print(lnode)#打出所有节点
#==============================================================================

#找u,v是邻居并有共同标签下的共同邻居z,计算与z有共同标签下的度
#==============================================================================
# common_neighbor={}
# for u in lnode_dict:
#     lnode1=lnode_dict[u]#u的标签
#     for v in lnode_dict:
#         lnode2=lnode_dict[v]#v的标签
#         if (g.has_edge(u,v)) and len(set(lnode1) & set(lnode2))>0:
# #            lnode=set(lnode1) & set(lnode2)
# #            print(u,v)#u,v为邻居节点
# #            print(lnode_dict[u],lnode_dict[v])#输出u,v的标签
# #            print(set(lnode1) & set(lnode2))#输出u,v共同标签
#             common_neighbors=nx.common_neighbors(g,u,v)
# common_neighbor=common_neighbors            
# for z in common_neighbor:#u,v共同邻居
# #    print(common_neighbor)
#     lnode3=lnode_dict[z]
#     neighbors=nx.neighbors(g,z)
#     for w in neighbors:#z的邻居
#         lnode4=lnode_dict[w]
#         if len(set(lnode3) & set(lnode4))>0:
#             degree=nx.degree(g,z)                        
#             print(degree)
#==============================================================================
#==============================================================================
# for w in neighbors:#z的邻居
#     lnode4=lnode_dict[w]
#     role_list=[]
#     if len(set(lnode3) & set(lnode4))>0:#与z有共同标签
#         triangle=role_list[z]
#         numerator = triangle + 1
#         degree=nx.degree(g,z)
#         non_triangle = degree * (degree - 1) / 2 - triangle
#         denominator = non_triangle + 1
#         role_list[z] = numerator / denominator            
#     print(degree)                
# 
#==============================================================================




##role_list = [nx.triangles(g, z) for z in range(node_num)]
#for u in lnode_dict:
#    lnode1=lnode_dict[u]#u的标签
#    for v in lnode_dict:
#        lnode2=lnode_dict[v]#v的标签
#        if (g.has_edge(u,v)) and len(set(lnode1) & set(lnode2))>0:
##            lnode=set(lnode1) & set(lnode2)
##            print(u,v)#u,v为邻居节点
##            print(lnode_dict[u],lnode_dict[v])#输出u,v的标签
##            print(set(lnode1) & set(lnode2))#输出u,v共同标签
#            common_neighbors=[nx.common_neighbors(g,u,v)]#u,v共同邻居
#            role_list = [nx.triangles(g, z) for z in range(node_num)]
#            print(role_list)
#            for z in common_neighbors:#u,v共同邻居
##            print(common_neighbor)
#                print(z)
#                lnode3=lnode_dict[z]
#                neighbors=nx.neighbors(g,z)#z的邻居                
##                print(neighbors)
#                for w in neighbors:#z的邻居
#                    lnode4=lnode_dict[z]
#                    
#                    if len(set(lnode3) & set(lnode4))>0:#与z有共同标签
#                        triangle=role_list[z]
#                        
#                        numerator = triangle + 1                        
#                        degree=nx.degree(g,z)
#                        non_triangle = degree * (degree - 1) / 2 - triangle
#                        denominator = non_triangle + 1
#                        role_list[z] = numerator / denominator            
##                print(degree) 
##                print('aa')



#==============================================================================
#     for u in lnode_dict:
#         lnode1=lnode_dict[u]#u的标签
#         for v in lnode_dict:
#             lnode2=lnode_dict[v]#v的标签
#             if (g.has_edge(u,v)) and len(set(lnode1) & set(lnode2))>0:
# #            lnode=set(lnode1) & set(lnode2)
# #            print(u,v)#u,v为邻居节点
# #            print(lnode_dict[u],lnode_dict[v])#输出u,v的标签
# #            print(set(lnode1) & set(lnode2))#输出u,v共同标签
#                 for z in nx.common_neighbors(g,u,v):#u,v共同邻居
# #            print(common_neighbor)
#                     lnode3=lnode_dict[z]
#                     neighbors=z | nx.neighbors(g,z)#z的邻居                
# #                print(neighbors)
#                     for w in neighbors:#z的邻居
#                         lnode4=lnode_dict[w]
#                         role_list=[]
#                         if len(set(lnode3) & set(lnode4))>0:#与z有共同标签
#                             triangle=role_list[z]
#                             numerator = triangle + 1
#                             degree=nx.degree(g,z)
#                             non_triangle = degree * (degree - 1) / 2 - triangle
#                             denominator = non_triangle + 1
#                             role_list[z] = numerator / denominator            
#                         print(degree)    
#==============================================================================
#role_list = [nx.triangles(g, z) for z in common_neighbors]   # 三角形个数
#print (role_list)
#for z in g:
#    triangle = role_list[z]
#    numerator = triangle + 1
#    d = nx.degree(g, z)
#    non_triangle = d * (d - 1) / 2 - triangle
#    denominator = non_triangle + 1
#    role_list[z] = numerator / denominator
#print(triangle)        









