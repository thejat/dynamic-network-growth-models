import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graph_generators import generate_fixed_group
from graph_estimators import estimate_lazy, estimate_bernoulli
import time, pickle, os, math
from collections import Counter

#helper functions
def localtime():
	return '_'.join([str(x) for x in time.localtime()[:5]])

def estimate_multiple_times(params,GT,glog=None):

	if params['dynamic']=='lazy':
		estimator = estimate_lazy
	elif params['dynamic']=='bernoulli':
		estimator = estimate_bernoulli
	else:
		 estimator = None
	assert estimator is not None

	estimates_dict = {}
	for t in params['estimation_indices']:
		print("  Estimating on sequence of length: ",t, " starting at time ", time.time()-params['start_time'])
		estimates_dict[t] = estimator(params,GT[:t],glog)
	return estimates_dict

def graph_stats_fixed_group(params,GT):
	nodecounts = []
	edgecounts = []
	for G in GT:
		nodecounts.append(len(G.nodes()))
		edgecounts.append(len(G.edges()))

	gtrue = {x[0]:x[1]['group'][0] for x in GT[0].nodes(data=True)} #only works for fixed group
	community_sizes = Counter([gtrue[i] for i in gtrue])

	return {'gtrue':gtrue, 'nodecounts': nodecounts, 'edgecounts': edgecounts}

def monte_carlo(params):

	np.random.seed()

	#Get graph sequence
	# print("Generate data: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	GT = generate_fixed_group(params['dynamic'],params['xitrue'],params['Mutrue'],params['Wtrue'],params['n'],params['k'],params['total_time'],params['start_time'])
	glog = graph_stats_fixed_group(params,GT)

	#Estimate parameters on each of the graphs at the given time indices
	# print("Estimate: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	log = estimate_multiple_times(params,GT,glog)
	print("\t   Run funish time:", time.time()-params['start_time'])

	return [log,glog]

def save_data(logs_glogs,params):
	params['end_time_delta'] 	= time.time() - params['start_time']
	fname = './output/pickles/log_'+params['dynamic']+'_'+localtime()+'.pkl'
	if params['only_unify'] is True:
		fname += '_'+params['unify_method']+'_unif' 
	pickle.dump({'log':[x for x,y in logs_glogs],'glog':[y for x,y in logs_glogs],'params':params},open(fname,'wb'))	
	print('Experiment end time:', params['end_time_delta'])	

def get_params():

	params 					= {}
	params['dynamic'] 		= 'bernoulli'
	params['n'] 			= 100 # size of the graph
	params['Mutrue'] 		= np.array([[.4,.6],[.6,.4]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.8,.2],[.2,.8]])
	params['k'] 			= params['Wtrue'].shape[0] # number of communities
	params['total_time'] 	= 32 # power of 2, number of additional graph snapshots
	params['nprocesses'] 	= 10
	params['n_mcruns'] 		= 2*params['nprocesses'] # number of monte carlo runs potentially in parallel [12 cores]
	params['estimation_indices'] = [int(math.pow(2,i)) for i in range(1,int(math.log2(params['total_time']))+1)]
	assert min(params['estimation_indices']) > 1
	params['xitrue'] 		= .2 # [lazy]
	params['ngridpoints']	= 21 # grid search parameter
	params['start_time'] 	= time.time()
	params['unify_method']  = 'sets' # 'lp' # 
	params['only_unify'] 	= False
	params['debug'] 		= False
	
	return params