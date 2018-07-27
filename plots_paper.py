import numpy as np
import pickle, pprint, os, time, glob
import seaborn as sns
from matplotlib import pyplot as plt
import pulp
from graph_estimators import error_between_scalars, error_between_matrices, error_between_groups

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



def get_permutation_from_LP(Q1,Qt):

	coeff = np.dot(np.transpose(Q1),Qt)
	
	tau = {}
	for i in range(Q1.shape[1]):
		for j in range(Q1.shape[1]):# Why both Q1.shape with the [1]?
			tau[(i,j)] = pulp.LpVariable("tau"+str(i)+str(j), 0, 1)

	lp_prob = pulp.LpProblem("Unify LP", pulp.LpMaximize)

	dot_cx = tau[(0,0)]*0
	for i in range(Q1.shape[1]):
		for j in range(Q1.shape[1]):
			dot_cx += tau[(i,j)]*coeff[i,j]
	lp_prob += dot_cx


	for i in range(Q1.shape[1]):
		constr = tau[(0,0)]*0
		for j in range(Q1.shape[1]):
			constr += tau[(i,j)]
		lp_prob += constr == 1

	for j in range(Q1.shape[1]):
		constr = tau[(0,0)]*0
		for i in range(Q1.shape[1]):
			constr += tau[(i,j)]
		lp_prob += constr == 1

	# lp_prob.writeLP('temp.lp')
	lp_prob.solve()

	tau = []
	for v in lp_prob.variables():
		# print "\t",v.name, "=", v.varValue
		tau.append(v.varValue)
	# print "\t Obj =", pulp.value(lp_prob.objective)
	return np.array(tau).reshape((Q1.shape[1],Q1.shape[1]))

def plot_error_vs_time(error,estimation_indices,error_std=None,attribute='',flag_write=False):
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
	plt.title(attribute)
	plt.show()
	if flag_write:
		fig.savefig('./output/'+'_'.join([str(x) for x in time.localtime()])+'.png', bbox_inches='tight', pad_inches=0.2)

def plot_fixed_group(fname,flag_write=False):
	rawdata = pickle.load(open(fname,'rb'))
	log, glog, params = rawdata['log'], rawdata['glog'], rawdata['params']

	attributes = {'wfinal':{'true_name':'Wtrue'},'gfinal':{'true_name':'gtrue'}}
	if params['dynamic']=='bernoulli':
		attributes['mufinal'] = {'true_name':'Mutrue'}
	elif params['dynamic']=='lazy':
		attributes['xifinal'] = {'true_name':'xitrue'}
	else:
		return NotImplementedError
	if params['only_unify'] is True:
		attributes = {'gfinal':{'true_name':'gtrue'}}
	
	def get_title(attribute,params):
		return attribute+' '+params['dynamic']+' n='+str(params['n'])+' k='+str(params['k'])+' '+params['unify_method']

	def get_tau(gtrue,gfinal):
		temp_nodes = gtrue.keys()
		k = max(gtrue.values())

		#Find permutation matrices tau
		Qtrue = np.zeros((len(temp_nodes),k))
		Qfinal = np.zeros((len(temp_nodes),k))
		for i in temp_nodes: #every node index from 1 to n
			Qtrue[i-1,gtrue[i]-1] = 1
			Qfinal[i-1,gfinal[i]-1] = 1

		tau = get_permutation_from_LP(Qtrue,Qfinal)

		return {'tau':tau, 'Qtrue':Qtrue, 'Qfinal':Qfinal}

	error = {}
	error_std = {}
	for attribute in attributes:
		error[attribute] = {}
		error_std[attribute] = {}
		for t in params['estimation_indices']:
			print('\n\n\n----------------------------\n\nt',t,'\n')
			temp = []
			for idx,x in enumerate(log):
				tau_info = get_tau(glog[idx]['gtrue'],x[t]['gfinal'])
				if attribute=='xifinal':
					temp.append(error_between_scalars(params[attributes[attribute]['true_name']],x[t][attribute]))
				elif attribute in ['wfinal','mufinal']:
					temp.append(error_between_matrices(params[attributes[attribute]['true_name']],x[t][attribute],attribute,tau_info))
				elif attribute=='gfinal':
					temp.append(error_between_groups(glog[idx]['gtrue'],x[t][attribute],tau_info))
				else:
					return NotImplementedError
			error[attribute][t] = np.mean(temp)
			error_std[attribute][t] = np.std(temp)

	for attribute in attributes:
		plot_error_vs_time(error[attribute],params['estimation_indices'],error_std[attribute],get_title(attribute,params),flag_write)

# def plot_individual(fname,flag_write=False):

# 	rawdata = pickle.load(open(fname,'rb'))
# 	log, glog, params = rawdata['log'], rawdata['glog'], rawdata['params']

# 	attributes = {'gfinal':{'true_name':'gtrue'},'wfinal':{'true_name':'Wtrue'}}
# 	if params['dynamic']=='bernoulli':
# 		attributes['mufinal'] = {'true_name':'Mutrue'}
# 	elif params['dynamic']=='lazy':
# 		attributes['xifinal'] = {'true_name':'xitrue'}
# 	else:
# 		return NotImplementedError
# 	if params['only_unify'] is True:
# 		attributes = {'gfinal':{'true_name':'gtrue'}}


# 	error = {}
# 	error_std = {}
# 	attribute = 'wfinal'
# 	for t in params['estimation_indices']:
# 		temp = []
# 		for idx,x in enumerate(log):
# 			tau_info = get_tau(glog[idx]['gtrue'],x[t]['gfinal'])
# 			if attribute=='xifinal':
# 				temp.append(error_between_scalars(params[attributes[attribute]['true_name']],x[t][attribute]))
# 			elif attribute in ['wfinal','mufinal']:
# 				temp.append(error_between_matrices(params[attributes[attribute]['true_name']],x[t][attribute],attribute,tau_info))
# 			elif attribute=='gfinal':
# 				temp.append(error_between_groups(glog[idx]['gtrue'],x[t][attribute],tau_info))
# 			else:
# 				return NotImplementedError
# 		error[[t] = np.mean(temp)
# 		error_std[t] = np.std(temp)

# 	plot_error_vs_time(error[attribute],params['estimation_indices'],error_std[attribute],get_title(attribute,params),flag_write)


if __name__ == '__main__':
	assert len(os.listdir('./output/pickles/')) is not None
	for fname in glob.glob('./output/pickles/*pkl*'):
		plot_fixed_group(fname,flag_write=True)