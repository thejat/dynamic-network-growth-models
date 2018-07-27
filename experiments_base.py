import os, matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graph_generators import generate_graph_sequence, add_noise, graph_stats_fixed_group
from graph_estimators import estimate_lazy, estimate_bernoulli
import time, pickle, os, math

#helper functions
def localtime():
	return '_'.join([str(x) for x in time.localtime()[2:5]])

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

def monte_carlo(params):

	np.random.seed()

	#Get graph sequence
	# print("Generate data: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	GT = generate_graph_sequence(params)	
	glog = graph_stats_fixed_group(params,GT)
	GTnoisy = GT
	if params['noisy_edges'] is True:
		GTnoisy = add_noise(GT)

	#Estimate parameters on each of the graphs at the given time indices
	# print("Estimate: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	log = estimate_multiple_times(params,GTnoisy,glog)
	print("\t   Run funish time:", time.time()-params['start_time'])

	return [log,glog]

def save_data(logs_glogs,params):
	params['end_time_delta'] 	= time.time() - params['start_time']
	fname = './output/pickles/log_'+params['dynamic']+'_n'+str(params['n'])+'_k'+str(params['k'])
	pickle.dump({'log':[x for x,y in logs_glogs],'glog':[y for x,y in logs_glogs],'params':params},open(fname+'_'+localtime()+'.pkl','wb'))	
	print('Experiment end time:', params['end_time_delta'])	

def get_params():

	params 					= {}
	params['dynamic'] 		= 'bernoulli'
	params['n'] 			= 100 # size of the graph
	params['Mutrue'] 		= np.array([[.4,.6],[.6,.4]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.4,.2],[.2,.4]])
	params['k'] 			= params['Wtrue'].shape[0] # number of communities
	params['total_time'] 	= 32 # power of 2, number of additional graph snapshots
	params['nprocesses'] 	= 10
	params['n_mcruns'] 		= params['nprocesses'] # number of monte carlo runs potentially in parallel [12 cores]
	params['estimation_indices'] = [int(math.pow(2,i))+1 for i in range(1,int(math.log2(params['total_time']))+1)]
	assert min(params['estimation_indices']) > 1
	params['xitrue'] 		= .5 # [lazy]
	params['ngridpoints']	= 21 # grid search parameter
	params['start_time'] 	= time.time()
	params['unify_method']  = 'UnifyCM' # 'UnifyLP' # 'Spectral-Mean'
	params['only_unify'] 	= False
	params['compare_unify'] = False
	params['debug'] 		= False
	params['noisy_edges'] 	= False
	params['spectral_adversarial'] = True
	params['minority_pct_ub'] = 0.2
	params['with_majority_dynamics'] = False
	
	return params