import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graph_generators import generate_fixed_group
from graph_estimators import estimate_lazy, estimate_bernoulli #EstimatorFixedGroupLazy, EstimatorFixedGroupBernoulli
import time, pickle
# from multiprocessing.pool import ThreadPool as Pool

#helper functions
def localtime():
	return '_'.join([str(x) for x in time.localtime()[:5]])

def estimate_multiple_times(params,GT,estimator=None):

	assert estimator is not None
	estimates_list =[]
	for t in params['estimation_indices']:
		if t < 1:
			return NotImplementedError #incorrect tbd
		else:
			print("  Estimating on sequence of length: ",t, " starting at time ", time.time()-params['start_time'])
			estimates = estimator(params,GT[:t])
		estimates_list.append(estimates)	
	return estimates



if __name__=='__main__':

	dynamic = 'bernoulli'
	# dynamic = 'lazy'

	#common parameters
	np.random.seed(1000)
	debug = False
	fname = './output/log_'+dynamic+'_'+localtime()+'.pkl'
	params = {}
	params['n_mcruns'] 		=   1 # number of monte carlo runs potentially in parallel
	params['total_time'] 	=   5 # number of additional graph snapshots
	params['estimation_indices'] = [params['total_time']]
	params['xitrue'] 		=   .2 # [lazy]
	params['Mutrue'] 		= np.array([[.5,.5],[.2,.6]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.8,.2],[.2,.8]])
	params['k'] 			= params['Wtrue'].shape[0] # number of communities
	params['n'] 			=   30# size of the graph
	params['ngridpoints']	=   21# grid search parameter
	params['start_time'] 	= time.time()
	params['processes'] 	= 10
	params['unify_method']  = 'lp' # 'sets' # 
	params['debug'] 		= False

	if dynamic=='lazy':
		estimator = estimate_lazy
	elif dynamic=='bernoulli':
		estimator = estimate_bernoulli
	else:
		 estimator = None


	#Get all graphs
	GTs = []

	# #pool version
	# gfg_arg_list = []
	# for mcrun in range(params['n_mcruns']):
	# 	gfg_arg_list.append(
	# 		(dynamic,params['xitrue'],params['Mutrue'],params['Wtrue'],params['n'],params['k'],params['total_time'],params['start_time'])
	# 		# {'dynamic':dynamic,
	# 		# 'xi':params['xitrue'],
	# 		# 'Mu': params['Mutrue'],
	# 		# 'W': params['Wtrue'],
	# 		# 'n': params['n'],
	# 		# 'k':params['k'],
	# 		# 'total_time': params['total_time'],
	# 		# 'start_time': params['start_time']}
	# 		)
	# pool = Pool(params['processes'])
	# for GT in pool.imap_unordered(generate_fixed_group, gfg_arg_list):
	# 	print("Generate data: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	# 	GTs.append(GT)


	for mcrun in range(params['n_mcruns']):
		print("Generate data: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])

		GTs.append(generate_fixed_group(dynamic,params['xitrue'],params['Mutrue'],params['Wtrue'],params['n'],params['k'],params['total_time'],params['start_time']))

	#Estimate parameters on each of the graphs at the given time indices
	log = {}
	for mcrun in range(params['n_mcruns']):
		print("Estimate: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
		log[mcrun] = estimate_multiple_times(params,GTs[mcrun],estimator)
		print("\t   Run funish time:", time.time()-params['start_time'])
		pickle.dump({'log':log,'params':params},open(fname,'wb'))
	
	print('Experiment end time:', time.time()-params['start_time'])
