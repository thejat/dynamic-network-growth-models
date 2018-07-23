from multiprocessing import Pool
from experiments_base import *

if __name__=='__main__':

	params = get_params() 	#common parameters

	# params['dynamic'] 	= 'lazy'
	# params['only_unify'] 	= True
	# params['unify_method']= 'lp'
	params['n'] 			= 100
	params['total_time']	= 16
	params['Mutrue'] 		= np.array([[.8,.2,.1,.1],[.2,.8,.2,.2],[.1,.2,.8,.2],[.1,.2,.2,.8]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.8,.2,.1,.1],[.2,.8,.2,.2],[.1,.2,.8,.2],[.1,.2,.2,.8]])
	params['k'] 			= params['Wtrue'].shape[0]

	with Pool(params['nprocesses']) as p:
		logs_glogs = p.map(monte_carlo,[params]*params['n_mcruns'])
	
	# print('DEBUG SEQUENTIAL')
	# logs_glogs = [monte_carlo(params),monte_carlo(params)]

	save_data(logs_glogs,params)