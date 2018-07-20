import numpy as np
import pickle,pprint
import seaborn as sns
from matplotlib import pyplot as plt

#Style
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['figure.titlesize'] = 30

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
	ts_errormeanw = np.zeros(params['total_time']-1)
	ts_errorstdw = np.zeros(params['total_time']-1)
	ts_errorw = np.zeros((params['total_time']-1,len(log)))
	ts_errorxi = np.zeros((params['total_time']-1,len(log)))
	ts_errormeanxi = np.zeros(params['total_time']-1)
	ts_errorstdxi = np.zeros(params['total_time']-1)
	ts_errorg = np.zeros((params['total_time']-1,len(log)))
	ts_errormeang = np.zeros(params['total_time']-1)
	ts_errorstdg = np.zeros(params['total_time']-1)
	for t in range(params['total_time']-1):
		for mcrun in range(len(log)):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanxi[t] += log[mcrun]['xifinals'][t]
			ts_errorw[t,mcrun] = np.linalg.norm(params['Wtrue']-log[mcrun]['wfinals'][t],'fro')
			ts_errorxi[t,mcrun] = abs(params['xitrue']-log[mcrun]['xifinals'][t])
		ts_meanw[t] = ts_meanw[t]*1.0/len(log)
		ts_meanxi[t] = ts_meanxi[t]*1.0/len(log)
	ts_errormeanw = np.mean(ts_errorw,axis=1)
	ts_errorstdw = np.std(ts_errorw,axis=1)
	ts_errormeanxi = np.mean(ts_errorxi,axis=1)
	ts_errorstdxi = np.std(ts_errorxi,axis=1)
	ts_errormeang = np.mean(ts_errorg,axis=1)
	ts_errorstdg = np.std(ts_errorg,axis=1)

	if debug:
		print 'actual runs: ', len(log)
		print 'mean w as a function of t'
		pprint.pprint(ts_meanw)
		print 'mean xi as a function of t'
		pprint.pprint(ts_meanxi)

	time = range(1,params['total_time'])
	title='FCLP-LazyRW: Error in the estimation of W'
	plot_error_vs_time(ts_errormeanw,time,title,ts_errorstdw,flag_write)
	title='FCLP-LazyRW: Error in the estimation of Xi'
	plot_error_vs_time(ts_errormeanxi,time,title,ts_errorstdxi,flag_write)
	title='FCLP-LazyRW: Estimation of Xi'
	plot_error_vs_time(ts_meanxi,time,title,flag_write=flag_write)
	# title='FCLP-LazyRW: Error in the estimation of Groups'
	# plot_error_vs_time(ts_errormeang,time,title,ts_errorstdg,flag_write)

def plot_fixed_bernoulli(fname,flag_write=False,debug=False):
	rawdata = pickle.load(open(fname,'rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_meanmu = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_errormeanw = np.zeros(params['total_time']-1)
	ts_errorstdw = np.zeros(params['total_time']-1)
	ts_errorw = np.zeros((params['total_time']-1,len(log)))
	ts_errormu = np.zeros((params['total_time']-1,len(log)))
	ts_errormeanmu = np.zeros(params['total_time']-1)
	ts_errorstdmu = np.zeros(params['total_time']-1)
	ts_errorg = np.zeros((params['total_time']-1,len(log)))
	ts_errormeang = np.zeros(params['total_time']-1)
	ts_errorstdg = np.zeros(params['total_time']-1)
	for t in range(params['total_time']-1):
		for mcrun in range(len(log)):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanmu[t] += log[mcrun]['mufinals'][t]
			ts_errorw[t,mcrun] = np.linalg.norm(params['Wtrue']-log[mcrun]['wfinals'][t],'fro')
			ts_errormu[t,mcrun] = np.linalg.norm(params['Mutrue']-log[mcrun]['mufinals'][t],'fro')

		ts_meanw[t] = ts_meanw[t]*1.0/len(log)
		ts_meanmu[t] = ts_meanmu[t]*1.0/len(log)

	ts_errormeanw = np.mean(ts_errorw,axis=1)
	ts_errorstdw = np.std(ts_errorw,axis=1)
	ts_errormeanmu = np.mean(ts_errormu,axis=1)
	ts_errorstdmu = np.std(ts_errormu,axis=1)
	ts_errormeang = np.mean(ts_errorg,axis=1)
	ts_errorstdg = np.std(ts_errorg,axis=1)

	if debug:
		pprint.pprint(ts_meanw)
		pprint.pprint(ts_meanmu)


	time = range(1,params['total_time'])
	title='FCLP-Bernoulli: Error in the estimation of W'
	plot_error_vs_time(ts_errormeanw,time,title,ts_errorstdw,flag_write)
	title='FCLP-Bernoulli: Error in the estimation of Mu'
	plot_error_vs_time(ts_errormeanmu,time,title,ts_errorstdmu,flag_write)
	# plot_error_vs_time([x[0,0] for x in ts_meanmu],time,title,flag_write=flag_write)
	# plot_error_vs_time([x[0,1] for x in ts_meanmu],time,title,flag_write=flag_write)
	# plot_error_vs_time([x[1,0] for x in ts_meanmu],time,title,flag_write=flag_write)
	# plot_error_vs_time([x[1,1] for x in ts_meanmu],time,title,flag_write=flag_write)
	# title='FCLP-Bernoulli: Error in the estimation of Groups'
	# plot_error_vs_time(ts_errormeang,time,title,ts_errorstdg,flag_write=flag_write)

def plot_changing_mm(fname,flag_write=False,debug=True):
	rawdata = pickle.load(open(fname,'rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_meanxi = np.zeros(params['total_time']-1)
	ts_errormeanw = np.zeros(params['total_time']-1)
	ts_errorstdw = np.zeros(params['total_time']-1)
	ts_errorw = np.zeros((params['total_time']-1,len(log)))
	ts_errorxi = np.zeros((params['total_time']-1,len(log)))
	ts_errormeanxi = np.zeros(params['total_time']-1)
	ts_errorstdxi = np.zeros(params['total_time']-1)

	for t in range(params['total_time']-1):
		for mcrun in range(len(log)):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanxi[t] += log[mcrun]['xifinals'][t]
			ts_errorw[t,mcrun] = np.linalg.norm(params['Wtrue']-log[mcrun]['wfinals'][t],'fro')
			ts_errorxi[t,mcrun] = abs(params['xitrue']-log[mcrun]['xifinals'][t])
		ts_meanw[t] = ts_meanw[t]*1.0/len(log)
		ts_meanxi[t] = ts_meanxi[t]*1.0/len(log)
	ts_errormeanw = np.mean(ts_errorw,axis=1)
	ts_errorstdw = np.std(ts_errorw,axis=1)
	ts_errormeanxi = np.mean(ts_errorxi,axis=1)
	ts_errorstdxi = np.std(ts_errorxi,axis=1)

	if debug:
		print 'actual runs: ', len(log)
		print 'mean w as a function of t'
		pprint.pprint(ts_meanw)
		print 'mean xi as a function of t'
		pprint.pprint(ts_meanxi)

	time = range(1,params['total_time'])
	title='CLP: Error in the estimation of W'
	plot_error_vs_time(ts_errormeanw,time,title,ts_errorstdw,flag_write)
	title='CLP: Error in the estimation of Xi'
	plot_error_vs_time(ts_errormeanxi,time,title,ts_errorstdxi,flag_write)
	title='CLP: Estimation of Xi'
	plot_error_vs_time(ts_meanxi,time,title,flag_write=flag_write)

if __name__ == '__main__':
	# plot_fixed_lazy('./output/explog_fixed_lazy.pkl',flag_write=False)
	# plot_fixed_bernoulli('./output/explog_fixed_bernoulli.pkl',flag_write=False)
	plot_changing_mm('./output/explog_changing_mm.pkl',flag_write=True)