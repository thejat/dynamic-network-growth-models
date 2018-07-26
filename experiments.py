from multiprocessing import Pool
from experiments_base import *

if __name__=='__main__':

	params = get_params() 	#common parameters

	params['dynamic'] 		= 'lazy'
	params['only_unify'] 	= True
	params['compare_unify'] = True
	params['n'] 			= 100
	params['total_time']	= 256
	params['estimation_indices'] = [int(math.pow(2,i))+1 for i in range(1,int(math.log2(params['total_time']))+1)]
	params['Mutrue'] 		= np.array([[.8,.2,.1,.1],[.2,.8,.2,.2],[.1,.2,.8,.2],[.1,.2,.2,.8]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.8,.2,.1,.1],[.2,.8,.2,.2],[.1,.2,.8,.2],[.1,.2,.2,.8]])
	# params['Wtrue'] 		= np.array([[.4,.2,.1,.1],[.2,.4,.2,.2],[.1,.2,.4,.2],[.1,.2,.2,.4]])
	params['k'] 			= params['Wtrue'].shape[0]


	with Pool(params['nprocesses']) as p:
		logs_glogs = p.map(monte_carlo,[params]*params['n_mcruns'])
	
	# logs_glogs = [monte_carlo(params),monte_carlo(params)] # debug without multiprocessing

	save_data(logs_glogs,params)