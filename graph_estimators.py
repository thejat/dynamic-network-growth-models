#graph parameter estimation
import numpy as np
import time, pprint, copy
import networkx as nx
from graph_tool import Graph, inference
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

def unify_communities_sets(ghats,k):

	Qs = {}
	QQtotal = np.zeros((len(ghats[0]),len(ghats[0])))
	for idx in range(len(ghats)):
		Qs[idx] = np.zeros((len(ghats[idx]),k))
		for i,x in enumerate(ghats[idx]):
			Qs[idx][i,ghats[idx][x]-1] = 1
		QQtotal += np.dot(Qs[idx],Qs[idx].transpose())

	spout = spectral_clustering(QQtotal,n_clusters=k)
	gfinal = {}
	for i in ghats[0]:
		gfinal[i] = spout[i-1]
	return gfinal

def get_communities_single_graph(G,k):
	def networkx2graph_tool(G):
		g = Graph()
		gv = {}
		ge = {}
		for n in G.nodes():
			gv[n] = g.add_vertex()
		for e in G.edges():
			ge[e] = g.add_edge(gv[e[0]],gv[e[1]])#connects two ends of the edges
		return [g,gv,ge]
	gtg,gtgv,gtge = networkx2graph_tool(G)
	gttemp = inference.minimize_blockmodel_dl(gtg,B_min=k,B_max=k)
	labels= np.array(gttemp.b.get_array())
	partition = {}
	for e,x in enumerate(gtgv):#Why for e and x?
		partition[x] = int(labels[e])+1 #Gives the # of nodes in each community?
	return partition

def get_communities_and_unify(params,GT):

	print('\t\tEstimating gfinal start: ',time.time()-params['start_time'])
	#Estimate communities for individual snapshots
	ghats = {}
	for t,G in enumerate(GT):
		ghats[t] = get_communities_single_graph(G,params['k'])

	#Unify
	if params['unify_method']=='sets':
		gfinal = unify_communities_sets(ghats,params['k'])
	elif params['unify_method']=='lp':
		gfinal = unify_communities_LP(ghats,params['k'])
	else:
		return NotImplementedError #incorrect tbd

	if params['debug']:
		for t in range(len(GT)):
			print('\tsnapshot',t)
			print('\t\t est:',ghats[t])
			print('\t\t truth:',{x[0]:x[1]['group'][0] for x  in GT[t].nodes(data=True)})

	print('\t\tEstimating gfinal end: ',time.time()-params['start_time'])
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

		def scoring(muvar, wvar, w_hats,r,s, GT):
			total = 0
			# print('t',t,' GT length is ',len(GT))
			for t in range(len(GT)+1):
				# print('t',t,' GT length is ',len(GT))
				total += np.power((np.power(1- muvar*(1+wvar),t)*wvar*wvar + wvar)/(1+wvar) - w_hats[t][r-1,s-1],2)
			return total

		grid_pts = np.linspace(0, 1, ngridpoints)
		muopt_array = []
		wopt_array = []
		score_log = np.zeros((len(grid_pts),len(grid_pts)))
		current_min = 1e8 #Potential bug
		muopt,wopt = grid_pts[0], grid_pts[0]
		for i,muvar in enumerate(grid_pts):
			for j,wvar in enumerate(grid_pts):
				candidate_score = scoring(muvar, wvar,w_hats,r,s,GT)
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


def get_communities_and_unify_debug_wrapper(params,GT,glog=None):

	# gfinal,ghats = get_communities_and_unify(params,GT) #ORIGINAL

	print('PASSING GTRUE **************************************** DEBUG')
	gfinal = glog['gtrue']
	ghats= {}

	########## debug
	compare=0
	if compare:
		print('Cross check the communities returned by sets and LP')
		params['unify_method'] 	= 'lp'
		gfinalLP,ghatsLP = get_communities_and_unify(params,GT)
		def get_group_error2(gtrue,gfinal):
			a,b = [0]*len(gtrue),[0]*len(gtrue)
			for i in gtrue:
				# print(i)
				a[i-1],b[i-1] = gtrue[i], gfinal[i]
			# return 1-metrics.adjusted_rand_score(a,b)
			return 1-metrics.adjusted_mutual_info_score(a,b)
		
		print("gfinal of sets and true differ: ",get_group_error2(glog['gtrue'],gfinal)," for t=",len(GT))
		print("gfinal of   lp and true differ: ",get_group_error2(glog['gtrue'],gfinalLP)," for t=",len(GT))
	########## debug

	return gfinal,ghats	

#Proposed Estimator for the Fixed Group Lazy Model 
def estimate_lazy(params,GT,glog=None):

	gfinal,ghats = get_communities_and_unify_debug_wrapper(params,GT,glog)
	if params['only_unify'] is True:
		return {'gfinal':gfinal,'ghats':ghats}
	w_hats = get_w_hats_at_each_timeindex(params,GT,gfinal)
	wfinal,xifinal = estimate_xi_and_w(params,GT,gfinal,w_hats)
	return {'gfinal':gfinal,'ghats':ghats,'wfinal':wfinal,'xifinal':xifinal}

#Proposed Estimator for the Fixed Group Bernoulli Model
def estimate_bernoulli(params,GT,glog=None):

	gfinal,ghats = get_communities_and_unify_debug_wrapper(params,GT,glog)

	if params['only_unify'] is True:
		return {'gfinal':gfinal,'ghats':ghats}
	w_hats = get_w_hats_at_each_timeindex(params,GT,gfinal)
	wfinal,mufinal = estimate_mu_and_w(params,GT,gfinal,w_hats)
	return {'gfinal':gfinal,'ghats':ghats,'wfinal':wfinal,'mufinal':mufinal}