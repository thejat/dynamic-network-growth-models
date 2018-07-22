import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graph_generators import generate_fixed_group
from graph_estimators import estimate_lazy, estimate_bernoulli
import time, pickle, os, math, argparse
from multiprocessing import Pool

#helper functions
def localtime():
	return '_'.join([str(x) for x in time.localtime()[:5]])

def estimate_multiple_times(params,GT):

	#decide which dynamic we are working with
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
		estimates_dict[t] = estimator(params,GT[:t])
	return estimates_dict

def graph_stats_fixed_group(params,GT):
	nodecounts = []
	edgecounts = []
	for G in GT:
		nodecounts.append(len(G.nodes()))
		edgecounts.append(len(G.edges()))

	gtrue = {x[0]:x[1]['group'][0] for x in GT[0].nodes(data=True)} #only works for fixed group

	return {'gtrue':gtrue, 'nodecounts': nodecounts, 'edgecounts': edgecounts}

def monte_carlo(params):

	np.random.seed()

	#Get graph sequence
	# print("Generate data: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	GT = generate_fixed_group(params['dynamic'],params['xitrue'],params['Mutrue'],params['Wtrue'],params['n'],params['k'],params['total_time'],params['start_time'])
	glog = graph_stats_fixed_group(params,GT)

	#Estimate parameters on each of the graphs at the given time indices
	# print("Estimate: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	log = estimate_multiple_times(params,GT)
	print("\t   Run funish time:", time.time()-params['start_time'])

	return [log,glog]


if __name__=='__main__':

	#common parameters
	params 	= {}
	GTs 	= []
	logs 	= []

	params['dynamic'] = 'bernoulli'
	params['n'] 			= 500 # size of the graph
	params['Mutrue'] 		= np.array([[.5,.5],[.2,.6]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.8,.2],[.2,.8]])
	params['k'] 			= params['Wtrue'].shape[0] # number of communities
	params['total_time'] 	= 32 # power of 2, number of additional graph snapshots

	parser = argparse.ArgumentParser()
	parser.add_argument("--dynamic", help="dynamic is either bernoulli (default) or lazy")
	parser.add_argument("--n", help="number of nodes (default 500)", type=int)
	parser.add_argument("--k", help="number of communities (default 2)", type=int)
	parser.add_argument("--t", help="number of additional time snaps (default 32)", type=int)
	args = parser.parse_args()

	if args.dynamic in ['bernoulli','lazy']:
		print(args.dynamic)
		params['dynamic'] = args.dynamic
	if args.n >= 32:
		print(args.n)
		params['n'] = args.n
	if args.k in [2,4]:
		print(args.k)
		if args.k == 4:
			params['k'] = args.k
			params['Mutrue'] 		= np.array([[.5,.5,.5,.5],[.2,.6,.2,.6],[.2,.5,.5,.5],[.2,.6,.2,.6]])# [bernoulli]
			params['Wtrue'] 		= np.array([[.8,.2,.1,.1],[.2,.8,.2,.2],[.1,.2,.8,.2],[.1,.2,.2,.8]])
	if args.t in [2,4,8,16,32,64,128]:
		params['total_time'] = args.t

	params['nprocesses'] 	= 10
	params['n_mcruns'] 		= params['nprocesses'] # number of monte carlo runs potentially in parallel [12 cores]
	params['estimation_indices'] = [int(math.pow(2,i)) for i in range(1,int(math.log2(params['total_time']))+1)]
	params['xitrue'] 		= .2 # [lazy]
	params['ngridpoints']	= 21 # grid search parameter
	params['start_time'] 	= time.time()
	params['unify_method']  = 'lp' # 'sets' # 
	params['debug'] 		= False
	assert min(params['estimation_indices']) > 1

	# for mcrun in range(params['n_mcruns']):
	# 	print("Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	# 	logs.append(monte_carlo(params))

	with Pool(params['nprocesses']) as p:
		logs_glogs = p.map(monte_carlo,[params]*params['n_mcruns'])


	params['end_time_delta'] 	= time.time() - params['start_time']
	pickle.dump({'log':[x for x,y in logs_glogs],'glog':[y for x,y in logs_glogs],'params':params},open('./output/pickles/log_'+params['dynamic']+'_'+localtime()+'.pkl','wb'))	
	print('Experiment end time:', params['end_time_delta'])