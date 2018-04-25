import os
import csv
import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import pdb
from collections import OrderedDict
import networkx as nx
import time
import string
from io import StringIO


# os.chdir("C:\\Users\\Mehrnaz\\Dropbox\\shared\\dynamic-network-growth")
#print os.getcwd()
#def generate_AS_graph(path):
csv_name = '201512.csv'

df = pd.read_csv(csv_name)

df['Column3'] = df['Column3'].astype(int)
df = df[df.Column3 != 0]

# pdb.set_trace()

st = time.time()
print 'Generating data'
Goriginal = nx.Graph()
nodes = set(df.Column1)
nodes = nodes.union(set(df.Column2))
n=len(nodes)
print 'n0:',n
temp_number = 0
contiguous = {}
for i in nodes:
    contiguous[i] = temp_number
    temp_number += 1

for i in nodes:
    Goriginal.add_node(contiguous[i])
    dest_i = df[df.Column1 == i]
    for j in set(dest_i.Column2):
        Goriginal.add_edge(contiguous[i], contiguous[j])
#print Goriginal.number_of_nodes()
#print Goriginal.number_of_edges()

#nx.draw(Goriginal, pos=nx.spring_layout(Goriginal), with_labels = True)

plt.show()
GT = [Goriginal]
t = 0
file_name = '20160{}.csv'
for i in range(1, 4):
    df=pd.read_csv(file_name.format(i))
    df['Column3'] = df['Column3'].astype(int)
    df = df[df.Column3 != 0]
    t = t + 1
    print '\tGraph at snapshot', t, ' time', time.time() - st

    Gcurrent = nx.Graph()
    nodes = set(df.Column1)
    nodes = nodes.union(set(df.Column2))
    n = len(nodes)
    print 'n of',t,n
    temp_number = 0
    contiguous = {}
    for i in nodes:
        contiguous[i] = temp_number
        temp_number += 1

    for i in nodes:
        Gcurrent.add_node(contiguous[i])
        dest_i = df[df.Column1 == i]
        for j in set(dest_i.Column2):
            Gcurrent.add_edge(contiguous[i], contiguous[j])

    GT.append(Gcurrent)
    print set(GT[t].nodes())-set(GT[t-1].nodes())
    print 'Inverse', set(GT[t-1].nodes())-set(GT[t].nodes())



    # nx.draw(Gcurrent, pos=nx.spring_layout(Gcurrent), with_labels=True, node_color=[ x[1]['group'] * 1.0 / 2 for x in Gcurrent.nodes(data=True) ])
    # plt.show()

print '\tTime taken:', time.time() - st

