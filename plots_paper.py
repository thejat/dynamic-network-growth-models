import numpy as np
import pickle,pprint

def plot_fixed_lazy():
	rawdata = pickle.load(open('explog_fixed_lazy.pkl','rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_meanxi = [0 for t in range(params['total_time']-1)]
	for t in range(params['total_time']-1):
		for mcrun in range(params['n_mcruns']):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanxi[t] += log[mcrun]['xifinals'][t]

		ts_meanw[t] = ts_meanw[t]*1.0/params['n_mcruns']
		ts_meanxi[t] = ts_meanxi[t]*1.0/params['n_mcruns']

	print 'mean w as a function of t'
	pprint.pprint(ts_meanw)
	print 'mean xi as a function of t'
	pprint.pprint(ts_meanxi)

def plot_fixed_bernoulli():
	rawdata = pickle.load(open('explog_fixed_bernoulli.pkl','rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_meanmu = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	for t in range(params['total_time']-1):
		for mcrun in range(params['n_mcruns']):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanmu[t] += log[mcrun]['mufinals'][t]

		ts_meanw[t] = ts_meanw[t]*1.0/params['n_mcruns']
		ts_meanmu[t] = ts_meanmu[t]*1.0/params['n_mcruns']

	pprint.pprint(ts_meanw)
	pprint.pprint(ts_meanmu)

def plot_changing_mm():
	rawdata = pickle.load(open('explog_changing_mm.pkl','rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	ts_meanxi = [0 for t in range(params['total_time']-1)]
	for t in range(params['total_time']-1):
		for mcrun in range(params['n_mcruns']):
			ts_meanw[t] += log[mcrun]['wfinals'][t]
			ts_meanxi[t] += log[mcrun]['xifinals'][t]

		ts_meanw[t] = ts_meanw[t]*1.0/params['n_mcruns']
		ts_meanxi[t] = ts_meanxi[t]*1.0/params['n_mcruns']

	print 'mean w as a function of t'
	pprint.pprint(ts_meanw)
	print 'mean xi as a function of t'
	pprint.pprint(ts_meanxi)

if __name__ == '__main__':
	plot_changing_mm()