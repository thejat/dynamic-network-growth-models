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

#os.chdir("C:\\Users\\Mehrnaz\\Dropbox\\shared\\dynamic-network-growth")
#print os.getcwd()
def generate_MIT_graph(path):
    csv_name = 'Calls.csv'

    df = pd.read_csv(csv_name)
    df_label= pd.read_csv('Subjects.csv')
    df_label['year_school']= df_label['year_school'].astype(str)

    df_label['year_school'] = df_label['year_school'].str.replace('Freshman', '1')
    df_label['year_school'] = df_label['year_school'].str.replace('Sophomore', '1')
    df_label['year_school'] = df_label['year_school'].str.replace('Senior', '1')
    df_label['year_school'] = df_label['year_school'].str.replace('Junior', '1')
    df_label['year_school'] = df_label['year_school'].str.replace('GRT / Other', '2')
    df_label['year_school'] = df_label['year_school'].str.replace('nan', '0')


    df_label['year_school']= df_label['year_school'].astype(int)

    #Create the group labels column

    #pdb.set_trace()
    df['label']=""
    for i in range(0, len(df_label.user_id)):
        id=df_label['user_id'][i]
        df.loc[df['user_id'] == id, 'label'] = df_label.year_school[i]


    df=df.dropna(subset=['user_id'])
    df=df.dropna(subset=['dest_user_id_if_known'])
    df=df[df.label != 0]

    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    datetime.datetime.fromtimestamp(1284286794)
    start_epoch = datetime.datetime(2008, 10, 1, 00, 00, 00)
    epoch=start_epoch

    next_date = datetime.datetime.strptime(str(epoch), '%Y-%m-%d %H:%M:%S').date()
    removal = df[df.time_stamp < next_date]
    remain = df[df.time_stamp > next_date]
    pre_date=next_date
    next_date = next_date + relativedelta(months=1)

    st = time.time()
    print 'Generating data'
    Goriginal = nx.Graph()
    n=len(df_label.user_id)
    temp_number = 0
    contiguous = {}
    for i in range(1, n+1):
        l= df_label['year_school'][i-1]
        if l == 1 or l==2:
            contiguous[i] = temp_number
            temp_number += 1

    for i in range(1, n+1):
        l= df_label['year_school'][i-1]
        if l==1:
            Goriginal.add_node(contiguous[i], group=np.random.choice(range(1, 2), 1))
        if l==2:
            Goriginal.add_node(contiguous[i], group=np.random.choice(range(2, 3), 1))
        else: pass
    for i in set(removal.user_id):
        dest_i = removal[removal.user_id == i]
        for j in set(dest_i.dest_user_id_if_known):
            if df_label['year_school'][j]!=0:
                Goriginal.add_edge(contiguous[i],contiguous[j])

    #print 'node color', [ x[1]['group'][0] * 1.0 / 2 for x in Goriginal.nodes(data=True) ]
    

    # nx.draw(Goriginal, pos=nx.spring_layout(Goriginal), with_labels = True, node_color=[ x[1]['group'][0] * 1.0 / 2 for x in Goriginal.nodes(data=True) ])
    # plt.show()

    GT = [Goriginal]
    t=0
    while len(remain) != 0:
        t=t+1
        print '\tGraph at snapshot', t, ' time', time.time() - st

        removal = remain[remain.time_stamp < next_date]
        remain = df[df.time_stamp > next_date]
        next_date = next_date + relativedelta(months=1)

        Gcurrent = nx.Graph()
        for node in GT[t - 1].nodes(data=True):
            Gcurrent.add_node(node[0], group=node[1]['group'])

        for i in set(removal.user_id):
            dest_i = removal[removal.user_id == i]
            for j in set(dest_i.dest_user_id_if_known):
                if df_label['year_school'][j] != 0:
                    Gcurrent.add_edge(contiguous[i],contiguous[j])

        GT.append(Gcurrent)



        # nx.draw(Gcurrent, pos=nx.spring_layout(Gcurrent), with_labels=True, node_color=[ x[1]['group'] * 1.0 / 2 for x in Gcurrent.nodes(data=True) ])
        # plt.show()

    print '\tTime taken:', time.time() - st
    return GT

if __name__=='__main__':
    # path = "C:\\Users\\Mehrnaz\\Dropbox\\shared\\dynamic-network-growth"
    path = './'
    GT = generate_MIT_graph(path)