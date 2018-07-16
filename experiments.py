import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graph_generators import generate_fixed_group
# from graph_estimators import EstimatorFixedGroupLazy, EstimatorFixedGroupBernoulli, EstimatorChangingGroupMM
import time, pickle


#helper functions
def localtime():
	return '_'.join([str(x) for x in time.localtime()[:5]])

def estimate(params):

	return None

	t_t 	  = []
	t_gfinal  = []
	t_wfinal  = []
	t_xifinal = [] # [lazy]
	t_mufinal = [] # [bernoulli]
	t_timing  = []
	for t in range(2,params['total_time']+1):
		print("  Estimating with number of snaps: ",t, " of", params['total_time'], ": starting at time", time.time()-params['start_time'])
		ghats,gfinal,w_hats,wfinal,xifinal,times = EstimatorFixedGroupLazy().estimate_params(GT[:t],params['k'],params['Wtrue'],params['ngridpoints']) #d
		t_gfinal.append(gfinal)
		t_wfinal.append(wfinal)
		t_xifinal.append(xifinal) #d
		t_t.append(t)
		t_timing.append(times)
	return {'graphs':GT,'gfinals':t_gfinal,'xifinals':t_xifinal,'n_snapshots':t_t,'wfinals':t_wfinal,'comptime':t_timing} #d



if __name__=='__main__':

	dynamic = 'bernoulli'
	# dynamic = 'lazy'

	#common parameters
	np.random.seed(1000)
	debug = False
	fname = './output/log_'+dynamic+'_'+localtime()+'.pkl'
	params = {}
	params['n_mcruns'] 		=   3 # number of monte carlo runs potentially in parallel
	params['total_time'] 	=   5 # number of additional graph snapshots
	params['estimation_indices'] = [params['total_time']]
	params['xitrue'] 		=   .2 # [lazy]
	params['Mutrue'] 		= np.array([[.5,.5],[.2,.6]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.8,.2],[.2,.8]])
	params['k'] 			= params['Wtrue'].shape[0] # number of communities
	params['n'] 			=   20# size of the graph
	params['ngridpoints']	=   21# grid search parameter
	params['start_time'] = time.time()

	#Get all graphs
	GTs = []
	for mcrun in range(params['n_mcruns']):
		print("Generate data: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])

		GTs.append(generate_fixed_group(dynamic,params['xitrue'],params['Mutrue'],params['Wtrue'],params['n'],params['k'],params['total_time']))

	#Estimate parameters on each of the graphs at the given time indices
	log = {}
	for mcrun in range(params['n_mcruns']):
		print("Estimate: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
		log[mcrun] = estimate(params)
		print("\t   Run funish time:", time.time()-params['start_time'])
		pickle.dump({'log':log,'params':params},open(fname,'wb'))
		print('Experiment end time:', time.time()-params['start_time'])
