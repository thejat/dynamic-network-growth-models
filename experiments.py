from multiprocessing import Pool
from experiments_base import *

if __name__=='__main__':

	params = get_params() 	#common parameters

	params['noisy'] 		= True
	params['dynamic'] 		= 'lazy'
	params['xitrue'] 		= 0.5 # [lazy]
	params['only_unify'] 	= True
	params['compare_unify'] = True
	# params['unify_method']= 'lp'
	params['n'] 			= 200
	# params['total_time']	= 128
	# params['estimation_indices'] = [int(math.pow(2,i))+1 for i in range(1,int(math.log2(params['total_time']))+1)]
	# params['estimation_indices'] = [int(math.pow(2,i))+1 for i in range(1,int(math.log2(params['total_time']))+1)][:2]
	params['Mutrue'] 		= np.array([[.8,.2,.1,.1],[.2,.8,.2,.2],[.1,.2,.8,.2],[.1,.2,.2,.8]])# [bernoulli]
	# params['Wtrue'] 		= np.array([[.4,.2,.1,.1],[.2,.4,.2,.2],[.1,.2,.4,.2],[.1,.2,.2,.4]])

	# np.random.seed(1000)
	# temp = np.random.rand(4,4)
	# params['Wtrue']			= 0.1*(temp + temp.transpose())
	# params['k'] 			= params['Wtrue'].shape[0]

	params['Wtrue'] 		= 0.5*np.array([[.4,.2,.1,.1],[.2,.4,.2,.2],[.1,.2,.4,.2],[.1,.2,.2,.4]])

	with Pool(params['nprocesses']) as p:
		logs_glogs = p.map(monte_carlo,[params]*params['n_mcruns'])
	
	# logs_glogs = [monte_carlo(params),monte_carlo(params)] # debug without multiprocessing

	save_data(logs_glogs,params)