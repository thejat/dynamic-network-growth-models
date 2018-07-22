from multiprocessing import Pool
from experiments_base import *

if __name__=='__main__':

	params = get_params() 	#common parameters

	params['dynamic'] = 'lazy'

	# params['n'] 		= 1000
	# params['total_time']= 32
	# params['Mutrue'] 	= np.array([[.5,.5,.5,.5],[.2,.6,.2,.6],[.2,.5,.5,.5],[.2,.6,.2,.6]])# [bernoulli]
	# params['Wtrue'] 	= np.array([[.8,.2,.1,.1],[.2,.8,.2,.2],[.1,.2,.8,.2],[.1,.2,.2,.8]])
	# params['k'] 		= params['Wtrue'].shape[0]

	with Pool(params['nprocesses']) as p:
		logs_glogs = p.map(monte_carlo,[params]*params['n_mcruns'])


	save_data(logs_glogs,params)