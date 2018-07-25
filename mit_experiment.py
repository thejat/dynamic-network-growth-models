import os, time, datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import networkx as nx
from experiments_base import *

def get_contiguous_ids(all_userids,df_label):
	userids_to_newids ={}
	newids_to_userids = {}
	tempidx = 1
	for user_id in all_userids:
		#hacky because index and user_id differ by 1 for df_label
		if df_label.year_school[user_id-1] > 0:
			userids_to_newids[user_id] = {'id':tempidx,'group':df_label.year_school[user_id-1]}
			newids_to_userids[tempidx] = user_id
			tempidx += 1

	return userids_to_newids,newids_to_userids

def construct_graph_sequence(df,userids_to_newids, newids_to_userids):

	st = time.time()
	print('Generating data')
	start_date = datetime.datetime(2008, 10, 1,0,0,0) #hard coded
	GT = []
	while True:
		df_current = df[df.date >= start_date]
		df_current = df_current[df_current.date < start_date + relativedelta(months=1)]
		print('start_date',start_date,' len(df_current) ',len(df_current))
		if len(df_current)==0:
			break
		else:
			Gcurrent = nx.Graph()
			for user_id in userids_to_newids:
				Gcurrent.add_node(userids_to_newids[user_id]['id'],group=np.array([userids_to_newids[user_id]['group']]))
			for user_id in userids_to_newids:
				df_temp = df_current.loc[df_current['user_id']==user_id,]
				#print(len(df_temp))
				for x in df_temp.dest_user_id_if_known:
					if x in userids_to_newids:
						Gcurrent.add_edge(userids_to_newids[user_id]['id'],userids_to_newids[x]['id'])
			GT.append(Gcurrent)
			print('new interval is',start_date,start_date + relativedelta(months=1))
			start_date = start_date + relativedelta(months=1)

	return GT

def processed_data():

	df = pd.read_csv('./data/Calls.csv')
	df_label= pd.read_csv('./data/Subjects.csv')
	# len(df) #63824
	df_label['year_school'] = df_label['year_school'].astype(str)
	df_label['year_school'] = df_label['year_school'].str.replace('Freshman', '1')
	df_label['year_school'] = df_label['year_school'].str.replace('Sophomore', '1')
	df_label['year_school'] = df_label['year_school'].str.replace('Senior', '1')
	df_label['year_school'] = df_label['year_school'].str.replace('Junior', '1')
	df_label['year_school'] = df_label['year_school'].str.replace('GRT / Other', '2')
	df_label['year_school'] = df_label['year_school'].str.replace('nan', '0')
	df_label['year_school'] = df_label['year_school'].astype(int)

	#drop NA values for nodes
	df=df.dropna(subset=['user_id'])
	df=df.dropna(subset=['dest_user_id_if_known'])

	#Change timestamps to dates
	df['date'] = pd.to_datetime(df['time_stamp']).dt.floor('d')
	df.index = range(len(df))

	return df,df_label

df,df_label = processed_data()

all_userids = set(df['user_id'].unique()).union(set(df['dest_user_id_if_known'].unique()))

userids_to_newids, newids_to_userids = get_contiguous_ids(all_userids,df_label) #for relabling

GT0 = construct_graph_sequence(df,userids_to_newids, newids_to_userids)

#Remove the last three months because they have lots of isolates
isolated = []
for G in GT0:
    isolated.append([x for x in nx.isolates(G)])
# print([len(x) for x in isolated])
isolated = isolated[:6]
# print([len(x) for x in isolated])

#Remove nodes that are isolated in at least one snapshot
beta_nodes = set()
for x in isolated:
    beta_nodes = set().union(beta_nodes,x)
print('beta_nodes',beta_nodes)
# print('len(beta_nodes)',len(beta_nodes))

beta_userids = [newids_to_userids[x] for x in beta_nodes]
all_userids_filtered = [x for x in all_userids if x not in beta_userids]
userids_to_newids_filtered, newids_to_userids_filtered = get_contiguous_ids(all_userids_filtered,df_label) #for relabling
GT = construct_graph_sequence(df,userids_to_newids_filtered, newids_to_userids_filtered)
GT = GT[:6]

#There will still be some isolated nodes, but less the better.




params 					= {}
params['dynamic'] 		= 'bernoulli'
params['n'] 			= len(GT[0])
params['k'] 			= 2 # number of communities
params['total_time'] 	= len(GT)
params['estimation_indices'] = range(2,len(GT))
params['ngridpoints']	= 21 # grid search parameter
params['start_time'] 	= time.time()
params['unify_method']  = 'sets' # 'lp' # 
params['only_unify'] 	= False
params['debug'] 		= False

glog = graph_stats_fixed_group(params,GT)
log = estimate_multiple_times(params,GT,glog)
print("\t   Run funish time:", time.time()-params['start_time'])

params['end_time_delta'] 	= time.time() - params['start_time']
fname = './output/pickles/real_mit_log_'+params['dynamic']+'_'+localtime()+'.pkl'
pickle.dump({'log':[log],'glog':[glog],'params':params, 'GT':GT},open(fname,'wb'))	
print('Experiment end time:', params['end_time_delta'])