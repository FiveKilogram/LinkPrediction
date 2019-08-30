# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:00:15 2017

@author: zhaosy
"""
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
import random 
def readfile(filename,w,c1,c2,pN,max_iter,v_max):
    G = nx.Graph(nx.read_pajek(filename))
    g=ig.Graph.Read_Pajek(filename)
    mmb = g.community_label_propagation().membership
    dim = g.vcount()           #搜索维度
    return g,pN,G,mmb,dim,w,c1,c2,max_iter,v_max
g,G,mmb,pN,dim,w,c1,c2,max_iter,v_max=readfile('karate.net',0.95,1.494,1.494,3,2,1)
def Q_mmb(G, mmb):    #### 2017.08.28
#  m=G.number_of_edges()
  m=g.ecount()
  e_iter = G.edges_iter
  deg={}; k={}

  for edge in e_iter():
    u, v = edge
    u = int(u)-1
    v = int(v)-1
    deg[mmb[u]]=1 if mmb[u] not in deg else deg[mmb[u]]+1
    deg[mmb[v]]=1 if mmb[v] not in deg else deg[mmb[v]]+1
    if mmb[u]==mmb[v]:
      k[mmb[u]]=1 if mmb[u] not in k else k[mmb[u]]+1
  #end of for e in e_iter()
  print (deg,k)
  Q = 0.0
  for i in k:
    Q += float(k[i]) -deg[i]/(2.0*m) * deg[i]/(2.0)
  Q /= m
  return Q

#g,pN,dim,w,c1,c2,max_iter,v_max=readfile('football.net',0.95,2,2,300,20,10)
#g,pN,dim,w,c1,c2,max_iter,v_max=readfile('dolphins.net',0.95,2,2,100,20,10)
#p =np.zeros((pN,dim))     #所有粒子的位置和速度
p = []
pa = []
v = []
vmax=v_max
vmin=-v_max
#vmax=[[v_max]*pN]*dim
#vmin=[[-v_max]*pN]*dim
fz = []
Pbestp=[]
Pbestq =[]
Gbestp=[]
Gbestq=[]
for i in range(3):
    p.append(g.community_label_propagation().membership)
    pa.append(p)
    v.append([random.uniform(-abs(vmax-vmin),abs(vmax-vmin)) for x in range(dim)])
    fz.append(g.modularity(p[i]))
#    fz.append(Q_mmb(G,p[i]))
#    pa.append(p[i])
    Pbestp.append(p[i])
    Pbestq.append(fz[i])
Gbestp=Pbestp[fz.index(max(fz))]
Gbestq=[max(fz),fz.index(max(fz))]
print (Gbestq)
#Gbestp.append(fz.index(max(fz)))
#Gbestq.append(max(fz))
#Gbestp.append(Pbestp[fz.index(max(fz))])
#Gbestq.append(max(fz),fz.index(max(fz)))
#    Gbestq=[max(fz),fz.index(max(fz))]
#Gbestq,Gbestp=