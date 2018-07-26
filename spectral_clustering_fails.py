from multiprocessing import Pool
from experiments_base import *
from graph_generators import generate_fixed_group_adversarial

def monte_carlo(params):

	np.random.seed()

	#Get graph sequence
	# print("Generate data: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	GT = generate_fixed_group_adversarial(params['dynamic'],params['xitrue'],params['Mutrue'],params['Wtrue'],params['n'],params['k'],params['total_time'],params['start_time'])	
	glog = graph_stats_fixed_group(params,GT)
	GTnoisy = GT
	if params['noisy'] is True:
		GTnoisy = add_noise(GT)

	#Estimate parameters on each of the graphs at the given time indices
	# print("Estimate: Monte Carlo Run # ",mcrun+1, " of ",params['n_mcruns'],' starting: ',time.time() - params['start_time'])
	log = estimate_multiple_times(params,GTnoisy,glog)
	print("\t   Run funish time:", time.time()-params['start_time'])

	return [log,glog]


if __name__=='__main__':

	params = get_params() 	#common parameters

	# params['noisy'] 		= True
	# params['dynamic'] 		= 'lazy'
	params['xitrue'] 		= 0.5 # [lazy]
	params['only_unify'] 	= True
	params['compare_unify'] = True
	# params['unify_method']= 'lp'
	params['n'] 			= 100
	params['total_time']	= 256
	params['estimation_indices'] = [int(math.pow(2,i))+1 for i in range(1,int(math.log2(params['total_time']))+1)]
	# params['estimation_indices'] = [int(math.pow(2,i))+1 for i in range(1,int(math.log2(params['total_time']))+1)][:2]
	params['Mutrue'] 		= np.array([[.8,.2,.1,.1],[.2,.8,.2,.2],[.1,.2,.8,.2],[.1,.2,.2,.8]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.8,.2,.1,.1],[.2,.8,.2,.2],[.1,.2,.8,.2],[.1,.2,.2,.8]])
	params['k'] 			= params['Wtrue'].shape[0]

	# np.random.seed(1000)
	# temp = np.random.rand(4,4)
	# params['Wtrue']			= 0.1*(temp + temp.transpose())
	# params['k'] 			= params['Wtrue'].shape[0]

	# params['Wtrue'] 		= 0.5*np.array([[.4,.2,.1,.1],[.2,.4,.2,.2],[.1,.2,.4,.2],[.1,.2,.2,.4]])

	with Pool(params['nprocesses']) as p:
		logs_glogs = p.map(monte_carlo,[params]*params['n_mcruns'])
	
	# logs_glogs = [monte_carlo(params),monte_carlo(params)] # debug without multiprocessing

	save_data(logs_glogs,params)
