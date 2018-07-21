import numpy as np
import pickle, pprint, os, time
import seaborn as sns
from matplotlib import pyplot as plt

#Style
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] 	= 'serif'
plt.rcParams['font.serif'] 		= 'Ubuntu'
plt.rcParams['font.monospace'] 	= 'Ubuntu Mono'
plt.rcParams['font.size'] 		= 30
plt.rcParams['axes.labelsize'] 	= 30
plt.rcParams['axes.titlesize'] 	= 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['figure.titlesize']= 30

def plot_error_vs_time(error,estimation_indices,error_std=None,flag_write=False):
	error = np.array([error[x] for x in error])
	error_std = np.array([error_std[x] for x in error_std])
	
	# print(estimation_indices)
	# print(error)
	# print(error_std)

	fig, ax = plt.subplots()
	ax.plot(estimation_indices,error)
	if error_std is not None:
		ax.fill_between(estimation_indices, error+error_std, error-error_std, color='yellow', alpha=0.5)
	plt.xlabel('Number of snapshots')
	plt.show()
	if flag_write:
		fig.savefig('./output/'+'_'.join([str(x) for x in time.localtime()])+'.png', bbox_inches='tight', pad_inches=0.2)

def plot_fixed_group(fname,flag_write=False):
	rawdata = pickle.load(open(fname,'rb'))
	log, params = rawdata['log'], rawdata['params']

	if params['dynamic']=='bernoulli':
		attributes = {'wfinal':{'size':(params['k'],params['k']),'true_name':'Wtrue'},'mufinal':{'size':(params['k'],params['k']),'true_name':'Mutrue'}}
	elif params['dynamic']=='lazy':
		attributes = {'wfinal':{'size':(params['k'],params['k']),'true_name':'Wtrue'},'xifinal':{'size':1,'true_name':'xitrue'}}
	else:
		return
	
	def err_between(a,b):
		return np.linalg.norm(a-b)

	error = {}
	error_std = {}
	for attribute in attributes:
		error[attribute] = {}
		error_std[attribute] = {}
		for t in params['estimation_indices']:
			temp = [err_between(params[attributes[attribute]['true_name']],x[t][attribute]) for x in log]
			error[attribute][t] = np.mean(temp)
			error_std[attribute][t] = np.std(temp)

	for attribute in attributes:
		plot_error_vs_time(error[attribute],params['estimation_indices'],error_std[attribute],flag_write)

if __name__ == '__main__':
	assert len(os.listdir('./output/pickles/')) is not None
	plot_fixed_group('./output/pickles/'+os.listdir('./output/pickles/')[0],flag_write=True)