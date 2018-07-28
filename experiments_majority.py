from graph_estimators import *
from experiments_base import *

np.random.seed(1000)

params 					= get_params()
params['n'] 			= 200 # size of the graph
params['Mutrue'] 		= np.array([[.4,.6],[.6,.4]])# [bernoulli]
params['Wtrue'] 		= np.array([[.9,.2],[.2,.9]])
params['k'] 			= params['Wtrue'].shape[0] # number of communities
params['xitrue'] 		= .2 # [lazy]
params['start_time'] 	= time.time()
params['spectral_adversarial'] = False
params['total_time'] 	=  4 # power of 2, number of additional graph snapshots
params['minority_pct_ub'] = 0.49
params['with_majority_dynamics'] = True
params['dynamic'] 		= 'lazy'
params['unify_method']  = 'Spectral-Mean'
params['compare_unify'] = False
params['only_unify']    = False
GT = generate_graph_sequence(params)
# #Diagnostics
# print([len(G.nodes()) for G in GT])
# for t in range(len(GT)-1): #minorities only matter upto the last but one graph
#     print(t,get_minority_nodes(GT[t]))
# print([x for x in GT[3].nodes(data=True) if x[1]['majority']==0])

assert len(GT) > 1 #there has to be a second graph to estimate minority labels for
# print('all nodes at ',0,GT[0].nodes)

GTmr = remove_minorities(GT)
GTsamesized,mapping_samesized = get_same_sized_graph_sequence(GTmr)
Gmajority = {}
Gminority = {}
gmajority = {}
gminority = {}
# for t in range(1,len(GT)-1): #for a t value, we have majorities upto t-1 and some nodes have made minority transitions at t

t = 1 #time index at which we want to estimate the labels of all nodes (some of which were minority in the previous)
# print('all nodes at ',t,GTmr[t-1].nodes())
# print('majority',[x[0] for x in GT[t-1].nodes(data=True) if x[1]['majority']==1])
# print('minority',[x[0] for x in GT[t-1].nodes(data=True) if x[1]['majority']==0])
Gmajority[t],Gminority[t] = create_two_SBM_graphs(GT,t)
assert len(Gminority[t]) > params['k']

gminority[t] = get_communities_single_graph_index_wrapper(Gminority[t],params['k'])
gmajority[t] = get_communities_single_graph_index_wrapper(Gmajority[t],params['k'])
# max([x for x in gminority[t]]+[x for x in gmajority[t]])
log = estimate_lazy(params,GTsamesized)
print(log['xifinal'])


gestimated = estimate_communities_including_minorities(params,GT,t,gmajority[t],gminority[t],log['xifinal'])
gtrue = {x[0]:x[1]['group'][0] for x in GT[t].nodes(data=True)}
# print(error_between_groups(gestimated,gtrue))
print(error_between_subsets(GT,t,gtrue,gestimated,'minority'))
print(error_between_subsets(GT,t,gtrue,gestimated,'majority'))
print(error_between_groups(gestimated,gtrue))