import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
np.random.seed(1000)
from graph_generators import generate_Zhang_modelA_modified, generate_fixed_group_lazy, generate_fixed_group_bernoulli, generate_changing_group_MM
from graph_estimators import EstimatorZhangAModified, EstimatorFixedGroupLazy, EstimatorFixedGroupBernoulli, EstimatorChangingGroupMM
import time, pickle

def run_experiment_Zhang_modelA_modified():

	#Legacy below

	# alphaTrue = 0.7
	# betaTrue = 0.4

	# st = time.time()
	# GT = generate_Zhang_modelA_modified(alpha=alphaTrue,beta=betaTrue,n0=20,flag_draw=False,total_time=1000,flag_arrivals=False)
	# print "generated data, time taken:",time.time() - st

	# st = time.time() 	
	# alpha,beta = estimate_random_dynamic_no_arrival_recursive(GT)
	# print "estimated from data, time taken: ",time.time() - st
	# print "True vals: alpha",alphaTrue," beta",betaTrue
	# print "Estimates: alpha",alpha," beta ",beta
	
	# st = time.time()
	# alpha,beta = estimate_random_dynamic_no_arrival_gridsearch(GT)
	# print "estimated from data, time taken: ",time.time() - st
	# print "True vals: alpha",alphaTrue," beta",betaTrue
	# print "Estimates: alpha",alpha," beta ",beta

	#Legacy above

	lmbdTrue = 0.7
	muTrue = 0.4

	GT = generate_Zhang_modelA_modified(alpha=lmbdTrue,beta=muTrue,n0=20,flag_draw=False,total_time=10,flag_arrivals=True)
	
	lmbd,mu = EstimatorZhangAModified().estimate_random_dynamic_with_arrival_recursive(GT)

	print "True vals: lambda",lmbdTrue," mu",muTrue
	print "Estimates: lambda",lmbd," mu ",mu

def run_experiment_fixed_group_lazy():
	debug = False
	params = {}
	params['n_mcruns'] 		=   5
	params['total_time'] 	=   16
	params['xitrue'] 		=   0
	params['Wtrue'] 		= np.array([[.65,.1],[.1,0.5]])#[[1,.0],[.0,1]])# #np.random.rand(k,k)
	params['k'] 			= params['Wtrue'].shape[0]
	params['n'] 			=  20
	start_time = time.time()

	def save_estimates(params):
		GT = generate_fixed_group_lazy(xi=params['xitrue'],W=params['Wtrue'],n=params['n'],k=params['k'],
							flag_draw=False,total_time=params['total_time'])
		t_t 	  = []
		t_gfinal  = []
		t_wfinal  = []
		t_xifinal = []
		t_timing  = []
		for t in range(2,params['total_time']+1):
			print "  Estimating with number of snaps: ",t, " of", params['total_time'], ": starting at time", time.time()-start_time
			ghats,gfinal,w_hats,wfinal,xifinal,times = EstimatorFixedGroupLazy().estimate_params(GT[:t],params['k'],params['Wtrue'])
			t_gfinal.append(gfinal)
			t_wfinal.append(wfinal)
			t_xifinal.append(xifinal)
			t_t.append(t)
			t_timing.append(times)
		return {'graphs':GT,'gfinals':t_gfinal,'xifinals':t_xifinal,'n_snapshots':t_t,'wfinals':t_wfinal,'comptime':t_timing}


	log = {}
	for mcrun in range(params['n_mcruns']):
		print "Estimation Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns']
		log[mcrun] = save_estimates(params)
		print "  Run funish time:", time.time()-start_time

		print 'Saving a log of the experiment. This will be overwritten.'
		pickle.dump({'log':log,'params':params},open('explog_fixed_lazy.pkl','wb'))
		print 'Experiment end time:', time.time()-start_time

def run_experiment_fixed_group_bernoulli():
	debug = False
	params = {}
	params['n_mcruns'] 		=   2
	params['total_time'] 	=  12
	params['Mutrue'] 		= np.array([[.5,.5],[.2,.6]])
	params['Wtrue'] 		= np.array([[.7,.1],[.1,.7]])#[[1,.0],[.0,1]])# #np.random.rand(k,k)
	params['k'] 			= params['Wtrue'].shape[0]
	params['n'] 			=  16
	start_time = time.time()

	def save_estimates(params):
		GT = generate_fixed_group_bernoulli(Mu = params['Mutrue'], W=params['Wtrue'], n=params['n'],k=params['k'],
		   					flag_draw=False, total_time = params['total_time'])
		t_t 	  = []
		t_gfinal  = []
		t_wfinal  = []
		t_mufinal = []
		t_timing  = []
		for t in range(2,params['total_time']+1):
			print "  Estimating with number of snaps: ",t, " of", params['total_time'], ": starting at time", time.time()-start_time
			ghats,gfinal,w_hats,wfinal,mufinal,times = EstimatorFixedGroupBernoulli().estimate_params(GT[:t],params['k'],params['Wtrue'],params['Mutrue'])
			t_gfinal.append(gfinal)
			t_wfinal.append(wfinal)
			t_mufinal.append(mufinal)
			t_t.append(t)
			t_timing.append(times)
		return {'graphs':GT,'gfinals':t_gfinal,'mufinals':t_mufinal,'n_snapshots':t_t,'wfinals':t_wfinal,'comptime':t_timing}


	log = {}
	for mcrun in range(params['n_mcruns']):
		print "Estimation Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns']
		log[mcrun] = save_estimates(params)
		print "  Run funish time:", time.time()-start_time

		print 'Saving a log of the experiment. This will be overwritten.'
		pickle.dump({'log':log,'params':params},open('explog_fixed_bernoulli.pkl','wb'))
		print 'Experiment end time:', time.time()-start_time

def run_experiment_changing_group_MM():
	debug = False
	params = {}
	params['n_mcruns'] 		=     5
	params['total_time'] 	=    15
	params['xitrue'] 		=    .4
	params['Wtrue'] 		= np.array([[.65,.1],[.1,0.65]])#[[1,.0],[.0,1]])
	params['k'] 			= params['Wtrue'].shape[0]
	params['n'] 			=    10
	params['minority_pct_ub'] = 0.2
	start_time = time.time()

	def save_estimates(params):
		GT = generate_changing_group_MM(minority_pct_ub=params['minority_pct_ub'],xi=params['xitrue'],W=params['Wtrue'],
				n=params['n'],k=params['k'], flag_draw=False,total_time=params['total_time'])
		t_t 	  = []
		t_gfinals = []
		t_mfinals = []
		t_wfinal  = []
		t_xifinal = []
		t_timing  = []
		for t in range(2,params['total_time']+1):
			print "  Estimating with number of snaps: ",t, " of", params['total_time'], ": starting at time", time.time()-start_time
			ghats,gfinals,mfinals,w_hats,wfinal,xifinal,times = EstimatorChangingGroupMM().estimate_params(GT[:t],params['k'],params['Wtrue'],params['xitrue'])

			t_gfinals.append(gfinals)
			t_mfinals.append(mfinals)
			t_wfinal.append(wfinal)
			t_xifinal.append(xifinal)
			t_t.append(t)
			t_timing.append(times)
		return {'graphs':GT,'gfinalst':t_gfinals,'mfinalst':t_mfinals,'xifinals':t_xifinal,'n_snapshots':t_t,'wfinals':t_wfinal,'comptime':t_timing}


	log = {}
	for mcrun in range(params['n_mcruns']):
		print "Estimation Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns']
		log[mcrun] = save_estimates(params)
		print "  Run funish time:", time.time()-start_time

		print 'Saving a log of the experiment. This will be overwritten.'
		pickle.dump({'log':log,'params':params},open('explog_changing_mm.pkl','wb'))
		print 'Experiment end time:', time.time()-start_time

if __name__=='__main__':

	#Zhang Model A Modified
	# run_experiment_Zhang_modelA_modified()

	#Fixed Group Lazy
	# run_experiment_fixed_group_lazy()

	#Fixed group Bernoulli
	# run_experiment_fixed_group_bernoulli()

	#Majority/Minority model
	run_experiment_changing_group_MM()