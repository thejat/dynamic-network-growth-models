#graph parameter estimation
import numpy as np
import scipy as sp
import time, pprint, copy
import networkx as nx
# from graph_tool import Graph, inference
import pulp #tbd: gurobipy/cplex
from sklearn.cluster import spectral_clustering
from sklearn import metrics

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

def unify_communities_LP(ghats,k):

	#Find permutation matrices tau's
	Qs = {}
	Qs[0] = np.zeros((len(ghats[0]),k))
	for i,x in enumerate(ghats[0]): #every node index in the first snapshot
		Qs[0][i,ghats[0][x]-1] = 1
	taus = {}
	for j in range(1,len(ghats)):#ghats except the first one

		#create the jth snapshot matrix
		Qs[j] = np.zeros((len(ghats[0]),k))
		for i,x in enumerate(ghats[0]): 
			if x in ghats[j]: #every node index in the jth snapshot
				Qs[j][i,ghats[j][x]-1] = 1					

		Qs_0_subset = np.zeros((len(ghats[0]),k))
		for i,x in enumerate(ghats[0]): #every node index in the first snapshot
			if x in ghats[j]: #every node index in the jth snapshot
				Qs_0_subset[i,ghats[0][x]-1] = 1

		taus[j] = get_permutation_from_LP(Qs_0_subset,Qs[j]) #always a k*k matrix

	#apply them on Qt's to get unpermuted group memberships
	gfinal = {}
	# print 'ghats[0] ', ghats[0]
	for i in ghats[0]:#for each node from 1 to n
		evec = np.zeros(len(ghats[0]))
		evec[i-1] = 1
		counts = np.dot(evec.transpose(),Qs[0])
		for l in range(1,len(ghats)):#for evert time index
			# print 'l',l,' eTQtau', np.dot(evec.transpose(),np.dot(Qs[l],np.linalg.inv(taus[l])))
			counts += np.dot(evec.transpose(),np.dot(Qs[l],np.linalg.inv(taus[l])))
		# print 'i',i,' counts',counts
		gfinal[i] = np.argmax(counts)+1


	return gfinal#, taus, Qs

def unify_communities_CM(ghats,k):

	Qs = {}
	QQtotal = np.zeros((len(ghats[0]),len(ghats[0])))
	for idx in range(len(ghats)):
		Qs[idx] = np.zeros((len(ghats[idx]),k))
		for i,x in enumerate(ghats[idx]):
			Qs[idx][i,ghats[idx][x]-1] = 1
		QQtotal += np.dot(Qs[idx],Qs[idx].transpose())

	spout = spectral_clustering(QQtotal,n_clusters=k) + 1
	gfinal = {}
	for i in ghats[0]:
		gfinal[i] = spout[i-1]
	return gfinal

def get_communities_single_graph(G,k):

	####### METHOD SPECTRAL
	spout = spectral_clustering(nx.adjacency_matrix(G),n_clusters=k) + 1
	gfinal = {}
	for i in G.nodes():
		gfinal[i] = spout[i-1]
	return gfinal

	# ####### METHOD BISECTION HEURISTIC https://arxiv.org/abs/1310.4378
	# def networkx2graph_tool(G):
	# 	g = Graph()
	# 	gv = {}
	# 	ge = {}
	# 	for n in G.nodes():
	# 		gv[n] = g.add_vertex()
	# 	for e in G.edges():
	# 		ge[e] = g.add_edge(gv[e[0]],gv[e[1]])#connects two ends of the edges
	# 	return [g,gv,ge]
	# gtg,gtgv,gtge = networkx2graph_tool(G)
	# gttemp = inference.minimize_blockmodel_dl(gtg,B_min=k,B_max=k)
	# labels= np.array(gttemp.b.get_array())
	# partition = {}
	# for e,x in enumerate(gtgv):#Why for e and x?
	# 	partition[x] = int(labels[e])+1 #Gives the # of nodes in each community?
	# return partition

def unify_communities_spectral_mean(params,GT):
	#This is the technique compared by Han Xu and Airoldi 2015 ICML (who propose variationam profile MLE algo)

	adj_matrix_summed = sp.sparse.csr_matrix(np.zeros((len(GT[0].nodes),len(GT[0].nodes))),dtype=int)
	for G in GT:
		adj_matrix_summed += nx.adjacency_matrix(G)

	spout = spectral_clustering(adj_matrix_summed,n_clusters=params['k']) + 1
	gfinal = {}
	for i in GT[0].nodes():
		gfinal[i] = spout[i-1]
	return gfinal,{}

def get_communities_and_unify(params,GT):

	#Unify by averaging over adjacency matrices
	if params['unify_method']=='Spectral-Mean':
		gfinal,ghats = unify_communities_spectral_mean(params,GT)
		return gfinal,ghats

	#If not doing the above, then unify after getting ghats

	# print('\t\tEstimating gfinal start: ',time.time()-params['start_time'])
	#First, estimate communities for individual snapshots
	ghats = {}
	for t,G in enumerate(GT):
		ghats[t] = get_communities_single_graph(G,params['k'])

	#Second, unify
	if params['unify_method']=='UnifyCM':
		gfinal = unify_communities_CM(ghats,params['k'])
	elif params['unify_method']=='UnifyLP':
		gfinal = unify_communities_LP(ghats,params['k'])
	else:
		return NotImplementedError #incorrect tbd

	if params['debug']:
		for t in range(len(GT)):
			print('\tsnapshot',t)
			print('\t\t est:',ghats[t])
			print('\t\t truth:',{x[0]:x[1]['group'][0] for x  in GT[t].nodes(data=True)})

	# print('\t\tEstimating gfinal end: ',time.time()-params['start_time'])
	return gfinal,ghats

def get_w_hats_at_each_timeindex(params,GT,gfinal):
	#Estimate w_hat_t_r_s for all t, r, and s in the sequence

	def estimate_w_mle(G,r,s,gfinal):
		
		rcount,scount,rscount = 0,0,0
		temp_nodes = G.nodes()
		temp_edges = G.edges()

		for x in temp_nodes:
			if gfinal[x]==r:
				rcount += 1
			if gfinal[x]==s:
				scount += 1
		if rcount<=0 or scount<=0:
			return 0

		for x in temp_nodes:
			for y in temp_nodes:
				if (gfinal[x] ==r and gfinal[y]==s) or (gfinal[x] ==s and gfinal[y]==r):
					if (x,y) in temp_edges or (y,x) in temp_edges:
						rscount += 1 #edge representations in networkx are directed

		# print(r,s,rcount,scount,rscount)
		if r==s:
			# in this case the mle is 2*number fo edges/((no of nodes)(no of nodes - 1)), 
			# but we are already double counting rscount above
			if scount == 1:
				return 0
			return rscount*1.0/(rcount*(scount - 1))
		else:
			return rscount*0.5/(rcount*scount)
	
	w_hats = {}
	k = params['k']

	for t,G in enumerate(GT):
		w_hats[t] = np.zeros((k,k))
		for r in range(1,k+1):
			for s in range(1,k+1):
				w_hats[t][r-1,s-1] = estimate_w_mle(G,r,s,gfinal) # gtruth # gfinal

	if params['debug']:
		for t in range(1,len(GT)+1):
			print('\n\t w_hats',t,w_hats[t-1])


	return w_hats

def estimate_mu_and_w(params,GT,gfinal,w_hats):

	def estimate_mu_and_w_given_r_s(w_hats, r, s, gfinal,GT, ngridpoints=21,debug=False):

		def scoring_equations(muvar, wvar, w_hats, r, s, GT, gfinal=None):
			total = 0
			# print('t',t,' GT length is ',len(GT))
			for t in range(len(GT)):
				# print('t',t,' GT length is ',len(GT))
				total += np.power((np.power(1- muvar*(1+wvar),t)*wvar*wvar + wvar)/(1+wvar) - w_hats[t][r-1,s-1],2)
			return total

		def scoring_mle(muvar, wvar, w_hats, r, s, GT, gfinal):
			#TBD doesn't seem to work correctly
			total = 0
			nodes = GT[0].nodes()
			for i in nodes:
				for j in nodes:
					if j > i: 
						if gfinal[i]==r and gfinal[j]==s:
							total += np.log(1e-20 + GT[0].has_edge(i,j)*wvar + (1-GT[0].has_edge(i,j))*(1-wvar))

							for t in range(1,len(GT)):
								total += np.log(1e-20 + (1-GT[t-1].has_edge(i,j))*GT[t].has_edge(i,j)*muvar*wvar) \
										+np.log(1e-20 + (1-GT[t-1].has_edge(i,j))*(1-GT[t].has_edge(i,j))*(1-muvar*wvar)) \
										+np.log(1e-20 + GT[t-1].has_edge(i,j)*GT[t].has_edge(i,j)*(1-muvar)) \
										+np.log(1e-20 + GT[t-1].has_edge(i,j)*(1-GT[t].has_edge(i,j))*muvar)


			return -1*total #minimize negative log likelihood

		grid_pts = np.linspace(0, 1, ngridpoints)
		muopt_array = []
		wopt_array = []
		score_log = np.zeros((len(grid_pts),len(grid_pts)))
		current_min = 1e8 #Potential bug
		muopt,wopt = grid_pts[0], grid_pts[0]
		for i,muvar in enumerate(grid_pts):
			for j,wvar in enumerate(grid_pts):
				candidate_score = scoring_mle(muvar, wvar,w_hats,r,s,GT,gfinal)
				# candidate_score = scoring_equations(muvar, wvar,w_hats,r,s,GT,gfinal)
				score_log[i,j] = candidate_score
				if np.isnan(candidate_score):
					continue
				if candidate_score <= current_min:
					muopt = muvar
					wopt = wvar
					current_min = candidate_score

		return muopt,wopt


	#Estimate wfinal and mufinal
	k = params['k']
	wfinal = np.zeros((k,k))
	mufinal = np.zeros((k,k))
	for r in range(1,k+1):
		for s in range(1,k+1):
			mufinal[r-1,s-1],wfinal[r-1,s-1] = estimate_mu_and_w_given_r_s(w_hats,r,s,gfinal,GT,params['ngridpoints'],debug=False)

	if params['debug']:
		print('\tmufinal', mufinal)
		print('\twfinal', wfinal)

	return wfinal,mufinal

def estimate_xi_and_w(params,GT,gfinal,w_hats):
		
	def estimate_w(w_hats,r,s):

		wopt_array = np.zeros(len(w_hats))
		for t in range(1,len(w_hats)+1):	
			wopt_array[t-1]= w_hats[t-1][r-1,s-1]
		return np.mean(wopt_array)

	def estimate_xi(wfinal,gfinal,GT,ngridpoints=21):

		def scoring(xivar,wfinal,gfinal,GT):
			score = 0
			for t in range(1,len(GT)):
				nodes = GT[t].nodes()
				temp_edges1 = GT[t].edges()
				temp_edges2 = GT[t-1].edges()
				for i in nodes:
					for j in nodes:
						if i < j:
							if (i,j) in temp_edges1:
								current_edge = 1
							else:
								current_edge = 0
							if (i,j) in temp_edges2:
								previous_edge = 1
							else:
								previous_edge = 0
							edge_copy = current_edge*previous_edge + (1-current_edge)*(1-previous_edge)

							score += np.log(1e-20 + xivar*edge_copy + (1-xivar)*(current_edge*wfinal[gfinal[i]-1,gfinal[j]-1] \
								+ (1-current_edge)*(1 - wfinal[gfinal[i]-1,gfinal[j]-1])))
			return score

		grid_pts = np.linspace(0,1,ngridpoints)
		current_max = scoring(grid_pts[0],wfinal,gfinal,GT)
		xiopt=0
		score_log = []
		for xivar in grid_pts:
			candidate_score = scoring(xivar,wfinal,gfinal,GT)
			score_log.append(candidate_score)
			if candidate_score >= current_max:
				xiopt = xivar
				current_max = candidate_score
		return xiopt


	k = params['k']

	#estimate w by relating w_hats_t to ws
	wfinal = np.zeros((k,k))
	for r in range(1,k+1):
		for s in range(1,k+1):
			wfinal[r-1,s-1] = estimate_w(w_hats,r,s)

	#estimate xi by 1-d grid search
	xifinal = estimate_xi(wfinal,gfinal,GT,params['ngridpoints'])


	if params['debug']:
		print('\twfinal', wfinal)
		print('\txifinal', xifinal)

	return wfinal,xifinal


def error_between_matrices(a,b,attribute,tau_info):
	tau 	= tau_info['tau']
	print(attribute)
	pprint.pprint(tau)
	pprint.pprint(a)
	pprint.pprint(b)
	# bnew = np.dot(np.dot(tau,b),tau) #this has some error tbd, is not needed with W and Mu are symmetric
	# pprint.pprint(bnew)
	print('np.linalg.norm(a-b): ',np.linalg.norm(a-b),'\t np.linalg.norm(a): ',np.linalg.norm(a))
	return np.linalg.norm(a-b)/np.linalg.norm(a)

def error_between_scalars(a,b):
	return np.abs(a-b)*1.0/a


def error_between_groups(gtrue,gfinal,tau_info=None):

	# #First type
	# assert tau_info is not None
	# tau 	= tau_info['tau']
	# Qtrue 	= tau_info['Qtrue']
	# Qfinal 	= tau_info['Qfinal']
	# return np.linalg.norm(Qtrue-np.dot(Qfinal,np.linalg.inv(tau)),'fro')*1.0/np.linalg.norm(Qtrue,'fro')

	#Second and third types
	a,b = [0]*len(gtrue),[0]*len(gtrue)
	for idx,i in enumerate(gtrue):
		a[idx],b[idx] = gtrue[i], gfinal[i]
	# return 1-metrics.adjusted_rand_score(a,b)
	return 1-metrics.adjusted_mutual_info_score(a,b)

def get_communities_and_unify_debug_wrapper(params,GT,glog=None):


	########## debug
	# print('PASSING GTRUE **************************************** DEBUG')
	# gfinal = glog['gtrue']
	# gfinal_menecessary{}
	########## debug


	gfinal_metadata = {}
	gfinal,ghats = get_communities_and_unify(params,GT) #necessary
	gfinal_metadata[params['unify_method']] = {'gfinal':gfinal,'ghats':ghats}

	if params['compare_unify']: #do the comparisons across different methods
		# print('Cross check the communities returned by UnifyCM and UnifyLP and Spectral-Mean')
		
		comparisons0 = set(['UnifyCM','UnifyLP','Spectral-Mean'])
		comparisons = comparisons0.difference([params['unify_method']])

		for method in comparisons:
			if method=='UnifyLP':
				params['unify_method'] 	= method
				gfinal2,ghats2 = get_communities_and_unify(params,GT)
				gfinal_metadata[method] = {'gfinal':gfinal2,'ghats':ghats2}
			elif method=='UnifyCM':
				params['unify_method'] 	= method
				gfinal2,ghats2 = get_communities_and_unify(params,GT)
				gfinal_metadata[method] = {'gfinal':gfinal2,'ghats':ghats2}
			elif method=='Spectral-Mean':
				params['unify_method'] 	= method
				gfinal2,ghats2 = get_communities_and_unify(params,GT)
				gfinal_metadata[method] = {'gfinal':gfinal2,'ghats':ghats2}

		for method in comparisons0:
			print("gfinal of ",method,"\t and true differ: ",error_between_groups(glog['gtrue'],gfinal_metadata[method]['gfinal'])," for t=",len(GT))

	return gfinal,gfinal_metadata

#Proposed Estimator for the Fixed Group Lazy Model 
def estimate_lazy(params,GT,glog=None):

	gfinal,gfinal_metadata = get_communities_and_unify_debug_wrapper(params,GT,glog)
	if params['only_unify'] is True:
		return {'gfinal':gfinal,'gfinal_metadata':gfinal_metadata}

	w_hats = get_w_hats_at_each_timeindex(params,GT,gfinal)
	wfinal,xifinal = estimate_xi_and_w(params,GT,gfinal,w_hats)
	return {'gfinal':gfinal,'gfinal_metadata':gfinal_metadata,'wfinal':wfinal,'xifinal':xifinal}

#Proposed Estimator for the Fixed Group Bernoulli Model
def estimate_bernoulli(params,GT,glog=None):

	gfinal,gfinal_metadata = get_communities_and_unify_debug_wrapper(params,GT,glog)

	if params['only_unify'] is True:
		return {'gfinal':gfinal,'gfinal_metadata':gfinal_metadata}

	w_hats = get_w_hats_at_each_timeindex(params,GT,gfinal)
	wfinal,mufinal = estimate_mu_and_w(params,GT,gfinal,w_hats)
	return {'gfinal':gfinal,'gfinal_metadata':gfinal_metadata,'wfinal':wfinal,'mufinal':mufinal}

def get_minority_nodes(G):
	temp_node_ids = []
	for x in G.nodes():
		if G.node[x]['majority']==0:
			temp_node_ids.append(x)
	return temp_node_ids

def remove_minorities(GT,till_time=None):
	'''
	if a node is minority at time s, then it is removed in the graph s+1 NOT in graph s
	'''

	assert len(GT) > 0

	if till_time is None:
		till_time = len(GT)-1
	else:
		assert len(GT) > till_time

	#The output sequence of graphs has lesser number of nodes across time
	GTmr = [GT[0]]

	for t in range(1, till_time+1):
		Gnew = GT[t].copy()
		minority_nodes = get_minority_nodes(GT[t-1])
		for i in minority_nodes:
			# print('removing node',i,'because it was a minority at time ',t-1)
			Gnew.remove_node(i)
		GTmr.append(Gnew)
	return GTmr

def get_new_node_ids(temp_node_ids):
	mapping = {'old2new': {}, 'new2old': {}}
	for idx,x in enumerate(temp_node_ids):
		mapping['old2new'][x] =idx+1 #node ids should always start from 1
		mapping['new2old'][idx+1] = x
	return mapping

def get_subgraph(Gcurrent,nodes):
	G = nx.Graph()
	for node in nodes:
		G.add_node(node)
	for node1 in nodes:
		for node2 in nodes:
			if Gcurrent.has_edge(node1,node2):
				G.add_edge(node1,node2)
	return G

def create_two_SBM_graphs(GT,t):
	#use the majority minority labels in Gprevious to create two graphs using edges in Gcurrent
	assert t > 0
	Gcurrent = GT[t]
	Gprevious = GT[t-1]
	if t==1:
		#we only care about new minorities
		minority_nodes = get_minority_nodes(Gprevious)
		new_minority_nodes = minority_nodes
	else:
		minority_nodes = get_minority_nodes(Gprevious)
		older_minority_nodes = get_minority_nodes(GT[t-2])
		new_minority_nodes = [x for x in minority_nodes if x not in older_minority_nodes]
	assert len(new_minority_nodes) > 1 #this is quite weak
	Gminority = get_subgraph(Gcurrent,new_minority_nodes)
	majority_nodes = [x for x in Gprevious.nodes() if x not in minority_nodes]
	Gmajority = get_subgraph(Gcurrent,majority_nodes)
	
	return Gmajority,Gminority


def get_communities_single_graph_index_wrapper(G,k):
	mapping = get_new_node_ids(G.nodes())
	Gtemp = nx.relabel_nodes(G,mapping['old2new'],copy=True)
	
	gtemp = get_communities_single_graph(Gtemp,k)
	
	gfinal = {}
	for new_node in gtemp:
		gfinal[mapping['new2old'][new_node]] = gtemp[new_node]

	return gfinal

def get_same_sized_graph_sequence(GTmr):
	GTsamesized = []
	retained_nodes= GTmr[-1].nodes()
	mapping = get_new_node_ids(retained_nodes)
	for G in GTmr:
		Gnew = G.copy()
		to_be_removed = [x for x in Gnew.nodes() if x not in retained_nodes]
		for node in to_be_removed:
			Gnew.remove_node(node)
		Gtemp = nx.relabel_nodes(Gnew,mapping['old2new'],copy=True)
		GTsamesized.append(Gtemp)
	return GTsamesized,mapping

def error_between_subsets(GT,t,gtrue,gestimated,subset_type='minority'):
	if subset_type=='minority':
		subset_nodes = [x[0] for x in GT[t-1].nodes(data=True) if x[1]['majority']==0]
	else:
		subset_nodes = [x[0] for x in GT[t-1].nodes(data=True) if x[1]['majority']==1]
		
	gtrue_subset = {}
	gestimated_subset = {}
	for x in subset_nodes:
		gtrue_subset[x] = gtrue[x]
		gestimated_subset[x] = gestimated[x]
	# print(gtrue_subset)
	# print(gestimated_subset)
	return error_between_groups(gtrue_subset,gestimated_subset)

def estimate_communities_including_minorities(params,GT,t,gmajority,gminority, xifinal):
	crosslinkmap,crosslinkmat = estimate_crosslink_freq(params,GT[t],gmajority,gminority,xifinal)
	# crosslinkmap = {1:1,2:2}
	print(crosslinkmat)
	print(crosslinkmap)
	gestimated = {}
	majority_nodes = [x[0] for x in GT[t-1].nodes(data=True) if x[1]['majority']==1]
	minority_nodes = [x[0] for x in GT[t-1].nodes(data=True) if x[1]['majority']==0]
	for node in GT[t].nodes():
		if node in majority_nodes:
			gestimated[node] = gmajority[node]
		else:
			gestimated[node] = crosslinkmap[gminority[node]]
	return gestimated

def estimate_crosslink_freq(params,G,gmajority,gminority):
	crosslinkmat = np.zeros((params['k'],params['k']))
	for r in range(1,params['k']+1):
		for s in range(1,params['k']+1):
			nodes1 = [x for x in gminority if gminority[x]==r]
			nodes2 = [x for x in gmajority if gmajority[x]==s]
			for node1 in nodes1:
				for node2 in nodes2:
					if G.has_edge(node1,node2) or G.has_edge(node2,node1):
						crosslinkmat[r-1,s-1] += 1
	crosslinkmap = {}
	if 1 - xifinal - xifinal*1.0/(params['k']-1) > 0:
		print('assign to highest crosslink')
		temp_map = np.argmax
	else:
		print('assign to lowest crosslink')
		temp_map = np.argmin

	for r in range(1,params['k']+1):
		crosslinkmap[r] = temp_map(crosslinkmat[r-1,:])+1
	return crosslinkmap,crosslinkmat
