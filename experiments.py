import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graph_generators import generate_fixed_group
from graph_estimators import estimate_lazy, estimate_bernoulli
import time, pickle
from multiprocessing.pool import ThreadPool as Pool

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


def monte_carlo(params):

	#Get graph sequence
	# print("Generate data: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	GT = generate_fixed_group(params['dynamic'],params['xitrue'],params['Mutrue'],params['Wtrue'],params['n'],params['k'],params['total_time'],params['start_time'])

	#Estimate parameters on each of the graphs at the given time indices
	# print("Estimate: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	log = estimate_multiple_times(params,GT)
	print("\t   Run funish time:", time.time()-params['start_time'])

	return log


if __name__=='__main__':

	#common parameters
	np.random.seed(1000)
	params = {}
	GTs = []
	log = []

	# params['dynamic'] = 'bernoulli'
	params['dynamic'] = 'lazy'
	params['n_mcruns'] 		=  10 # number of monte carlo runs potentially in parallel
	params['total_time'] 	=  10 # number of additional graph snapshots
	params['estimation_indices'] = [2,4,6,8,params['total_time']]
	params['xitrue'] 		=   .2 # [lazy]
	params['Mutrue'] 		= np.array([[.5,.5],[.2,.6]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.8,.2],[.2,.8]])
	params['k'] 			= params['Wtrue'].shape[0] # number of communities
	params['n'] 			=   30# size of the graph
	params['ngridpoints']	=   21# grid search parameter
	params['start_time'] 	= time.time()
	params['nprocesses'] 	= 32
	params['unify_method']  = 'lp' # 'sets' # 
	params['debug'] 		= False
	assert min(params['estimation_indices']) > 1

	# for mcrun in range(params['n_mcruns']):
	# 	print("Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	# 	log[mcrun] = monte_carlo(params)

	with Pool(params['nprocesses']) as p:
		log = p.map(monte_carlo,[params]*params['n_mcruns'])


	pickle.dump({'log':log,'params':params},open('./output/pickles/log_'+params['dynamic']+'_'+localtime()+'.pkl','wb'))	
	print('Experiment end time:', time.time()-params['start_time'])
