import numpy as np
import pickle,pprint
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12


def plot_error_vs_time(error,time,title):

	fig, ax = plt.subplots()
	ax.plot(time,error)
	ax.set_title(title)
	plt.show()


def plot_fixed_lazy():
	rawdata = pickle.load(open('explog_fixed_lazy.pkl','rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_meanxi = [0 for t in range(params['total_time']-1)]
	for t in range(params['total_time']-1):
		for mcrun in range(len(log)):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanxi[t] += log[mcrun]['xifinals'][t]

		ts_meanw[t] = ts_meanw[t]*1.0/params['n_mcruns']
		ts_meanxi[t] = ts_meanxi[t]*1.0/params['n_mcruns']

	print 'mean w as a function of t'
	pprint.pprint(ts_meanw)
	print 'mean xi as a function of t'
	pprint.pprint(ts_meanxi)

	error = [np.linalg.norm(params['Wtrue']-x,'fro') for x in ts_meanw]
	time = range(1,params['total_time'])
	title='Estimation of W'
	plot_error_vs_time(error,time,title)
	error = [abs(params['xitrue']-x) for x in ts_meanxi]
	time = range(1,params['total_time'])
	title='Estimation of Xi'
	plot_error_vs_time(error,time,title)

def plot_fixed_bernoulli():
	rawdata = pickle.load(open('explog_fixed_bernoulli.pkl','rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_meanmu = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	for t in range(params['total_time']-1):
		for mcrun in range(len(log)):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanmu[t] += log[mcrun]['mufinals'][t]

		ts_meanw[t] = ts_meanw[t]*1.0/params['n_mcruns']
		ts_meanmu[t] = ts_meanmu[t]*1.0/params['n_mcruns']

	pprint.pprint(ts_meanw)
	pprint.pprint(ts_meanmu)

	error = [np.linalg.norm(params['Wtrue']-x,'fro') for x in ts_meanw]
	time = range(1,params['total_time'])
	title='Estimation of W'
	plot_error_vs_time(error,time,title)
	error = [np.linalg.norm(params['Mutrue']-x,'fro') for x in ts_meanmu]
	time = range(1,params['total_time'])
	title='Estimation of Mu'
	plot_error_vs_time(error,time,title)

def plot_changing_mm():
	rawdata = pickle.load(open('explog_changing_mm.pkl','rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_meanxi = [0 for t in range(params['total_time']-1)]
	for t in range(params['total_time']-1):
		for mcrun in range(len(log)):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanxi[t] += log[mcrun]['xifinals'][t]

		ts_meanw[t] = ts_meanw[t]*1.0/params['n_mcruns']
		ts_meanxi[t] = ts_meanxi[t]*1.0/params['n_mcruns']

	print 'mean w as a function of t'
	pprint.pprint(ts_meanw)
	print 'mean xi as a function of t'
	pprint.pprint(ts_meanxi)
	
	error = [np.linalg.norm(params['Wtrue']-x,'fro') for x in ts_meanw]
	time = range(1,params['total_time'])
	title='Estimation of W'
	plot_error_vs_time(error,time,title)
	error = [abs(params['xitrue']-x) for x in ts_meanxi]
	time = range(1,params['total_time'])
	title='Estimation of Xi'
	plot_error_vs_time(error,time,title)

if __name__ == '__main__':
	# plot_fixed_lazy()
	plot_fixed_bernoulli()
	# plot_changing_mm()