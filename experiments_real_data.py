import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
np.random.seed(1000)
from graph_estimators import EstimatorZhangAModified, EstimatorFixedGroupLazy, EstimatorFixedGroupBernoulli, EstimatorChangingGroupMM
import time, pickle
import MITData

import pickle,pprint
import seaborn as sns
from matplotlib import pyplot as plt
from graph_estimators import EstimatorFixedGroupLazy

plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 30
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['figure.titlesize'] = 30



def run_experiment_fixed_group_lazy(fname):

	GT = MITData.generate_MIT_graph('./')

	debug = False
	params = {}
	params['n_mcruns'] 		=   1
	params['total_time'] 	=   len(GT)
	params['Wtrue'] 		= np.array([[0,0],[0,0]])#[[1,.0],[.0,1]])# #np.random.rand(k,k)
	params['k'] 			=   2
	params['n'] 			=   len(GT[0].nodes())
	params['ngridpoints']	=   21
	start_time = time.time()
	log = {}

	t_t 	  = []
	t_gfinal  = []
	t_wfinal  = []
	t_xifinal = []
	t_timing  = []
	for t in range(2,params['total_time']+1):
		print "  Estimating with number of snaps: ",t, " of", params['total_time'], ": starting at time", time.time()-start_time
		ghats,gfinal,w_hats,wfinal,xifinal,times = EstimatorFixedGroupLazy().estimate_params(GT[:t],params['k'],params['Wtrue'],params['ngridpoints'])
		t_gfinal.append(gfinal)
		t_wfinal.append(wfinal)
		t_xifinal.append(xifinal)
		t_t.append(t)
		t_timing.append(times)
	log[0] =  {'graphs':GT,'gfinals':t_gfinal,'xifinals':t_xifinal,'n_snapshots':t_t,'wfinals':t_wfinal,'comptime':t_timing}

	print "  Run funish time:", time.time()-start_time
	print 'Saving a log of the experiment. This will be overwritten.'
	pickle.dump({'log':log,'params':params},open(fname,'wb'))
	print 'Experiment end time:', time.time()-start_time

def run_experiment_fixed_group_bernoulli(fname):
	debug = False
	params = {}
	params['n_mcruns'] 		=  10
	params['total_time'] 	=  30
	params['Mutrue'] 		= np.array([[.5,.5],[.2,.6]])
	params['Wtrue'] 		= np.array([[.7,.1],[.1,.7]])#[[1,.0],[.0,1]])# #np.random.rand(k,k)
	params['k'] 			= params['Wtrue'].shape[0]
	params['n'] 			= 100
	params['ngridpoints']	=  41
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
			ghats,gfinal,w_hats,wfinal,mufinal,times = EstimatorFixedGroupBernoulli().estimate_params(GT[:t],params['k'],params['Wtrue'],params['Mutrue'],params['ngridpoints'])
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
		pickle.dump({'log':log,'params':params},open(fname,'wb'))
		print 'Experiment end time:', time.time()-start_time


def plot_error_vs_time(error,time,title,errorstd=None,flag_write=False):

	fig, ax = plt.subplots()
	ax.plot(time,error)
	if errorstd is not None:
		ax.fill_between(time, error+errorstd, error-errorstd, color='yellow', alpha=0.5)
	ax.set_title(title)
	plt.ylabel('Error')
	plt.xlabel('Number of snapshots')
	plt.show()
	if flag_write:
		fig.savefig('./output/'+title.replace(':','').replace('-','').replace(' ','_')+'.png', bbox_inches='tight', pad_inches=0.2)

def plot_fixed_lazy(fname,flag_write=False,debug=True):
	rawdata = pickle.load(open(fname,'rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_meanxi = np.zeros(params['total_time']-1)
	ts_errorg = np.zeros((params['total_time']-1,len(log)))
	ts_errormeang = np.zeros(params['total_time']-1)
	ts_errorstdg = np.zeros(params['total_time']-1)
	for t in range(params['total_time']-1):
		for mcrun in range(len(log)):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanxi[t] += log[mcrun]['xifinals'][t]
			ts_errorg[t,mcrun] = EstimatorFixedGroupLazy().get_group_error(log[mcrun]['graphs'][0],log[mcrun]['gfinals'][t],params['k'],debug=False)
		ts_meanw[t] = ts_meanw[t]*1.0/len(log)
		ts_meanxi[t] = ts_meanxi[t]*1.0/len(log)
	ts_errormeang = np.mean(ts_errorg,axis=1)
	ts_errorstdg = np.std(ts_errorg,axis=1)

	if debug:
		print 'actual runs: ', len(log)
		print 'mean w as a function of t'
		pprint.pprint(ts_meanw)
		print 'mean xi as a function of t'
		pprint.pprint(ts_meanxi)
		print 'estimated groups vs true'
		pprint.pprint([(x[0],log[mcrun]['gfinals'][8][x[0]],x[1]['group'][0]) for x  in log[mcrun]['graphs'][0].nodes(data=True)])

	time = range(1,params['total_time'])
	title='Estimation of W00'
	plot_error_vs_time([x[0,0] for x in ts_meanw],time,title,flag_write=flag_write)
	title='Estimation of Xi'
	plot_error_vs_time(ts_meanxi,time,title,flag_write=flag_write)
	title='Error in the estimation of Groups'
	plot_error_vs_time(ts_errormeang,time,title,ts_errorstdg,flag_write)



if __name__=='__main__':

	#Fixed Group Lazy
	# run_experiment_fixed_group_lazy('./output/explog_fixed_lazy.pkl')

	# rawdata = pickle.load(open('./output/explog_fixed_lazy.pkl','rb'))
	# log= rawdata['log']
	plot_fixed_lazy('./output/explog_fixed_lazy.pkl',flag_write=False)


	#Fixed group Bernoulli
	# run_experiment_fixed_group_bernoulli('./output/explog_fixed_bernoulli.pkl')

	# GT = MITData.generate_MIT_graph('./')