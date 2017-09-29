#graph parameter estimation
import numpy as np
import time
import networkx as nx
# import community
from graph_tool import Graph, collection, inference
import pulp #tbd: gurobipy/cplex
np.random.seed(1000)

#Common Functions
class EstimatorUtility(object):

	def networkx2graph_tool(self,G):
		g = Graph()
		gv = {}
		ge = {}
		for n in G.nodes():
			gv[n] = g.add_vertex()
		for e in G.edges():
			ge[e] = g.add_edge(gv[e[0]],gv[e[1]])
		return [g,gv,ge]

	def graph_tool_community(self,G,k):
		gtg,gtgv,gtge = self.networkx2graph_tool(G)
		gttemp = inference.minimize_blockmodel_dl(gtg,B_min=k,B_max=k)
		labels= np.array(gttemp.b.get_array())
		partition = {}
		for e,x in enumerate(gtgv):
			partition[x] = int(labels[e])+1
		return partition

#Proposed Estimator for the Fixed Group Lazy Model 
class EstimatorFixedGroupLazy(object):


	def get_permutation_from_LP(self,Q1,Qt):

		coeff = np.dot(np.transpose(Q1),Qt)
		
		tau = {}
		for i in range(Q1.shape[1]):
			for j in range(Q1.shape[1]):
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

	def unify_communities_LP(self,ghats,k):

		#Find permutation matrices tau's
		Qs = {}
		for j in ghats:#0,1.,...,T
			Qs[j] = np.zeros((len(ghats[0]),k))
			for i in range(1,len(ghats[0])+1): #every node index from 1 to n
				Qs[j][i-1,ghats[j][i]-1] = 1

		taus = {}
		for j in range(1,len(ghats)):#ghats except the first one
			taus[j] = self.get_permutation_from_LP(Qs[0],Qs[j])

		#apply them on Qt's to get unpermuted group memberships
		gfinal = {}
		for i in range(1,len(ghats[0])+1):#for each node from 1 to n
			evec = np.zeros(len(ghats[0]))
			evec[i-1] = 1
			counts = np.dot(evec.transpose(),Qs[0])
			for l in range(1,len(ghats)):#for evert time index
				# print 'l',l,' eTQtau', np.dot(evec.transpose(),np.dot(Qs[l],np.linalg.inv(taus[l])))
				counts += np.dot(evec.transpose(),np.dot(Qs[l],np.linalg.inv(taus[l])))
			# print 'i',i,' counts',counts
			gfinal[i] = np.argmax(counts)+1


		return gfinal#, taus, Qs



	def unify_communities_sets(self,ghats,k):
		kappa = 1
		gfinal = {}
		countij = {}
		for i in ghats[0].keys():
			for j in range(1,i):
				countij[(i,j)] = 0
				for t in range(0,len(ghats)):
					if ghats[t][i]==ghats[t][j]:
						countij[(i,j)] += 1
				if countij[(i,j)] > len(ghats)*0.5:#BUG
					if gfinal.get(j,None) is not None and gfinal.get(i,None) is None:
						gfinal[i] = gfinal[j]
					if gfinal.get(i,None) is None and gfinal.get(j,None) is None:
						gfinal[i] = kappa
						gfinal[j] = kappa
						kappa += 1
		return gfinal

	def estimate_w_mle(self,G,r,s,gfinal,debug=True):
		rcount,scount,rscount = 0,0,0

		# if debug:
		# 	print 'gfinal',gfinal

		for x in G.nodes():
			if gfinal[x]==r:
				rcount += 1
			if gfinal[x]==s:
				scount += 1

		for x in G.nodes():
			for y in G.nodes():
				if (gfinal[x] ==r and gfinal[y]==s) or (gfinal[x] ==s and gfinal[y]==r):
					if (x,y) in G.edges() or (y,x) in G.edges():
						rscount += 1 #edge representations in networkx are directed

		# if debug:
		# 	print r,s,rcount,scount,rscount

		if rcount==0 or scount==0:
			return 0
		else:
			return rscount*1.0/(rcount*scount)

	def xiw_model_estimate_w(self,w_hats,r,s,debug=False):

		wopt_array = np.zeros(len(w_hats))
		for t in range(1,len(w_hats)+1):	
			wopt_array[t-1]= w_hats[t-1][r-1,s-1]

		print wopt_array,np.mean(wopt_array),np.median(wopt_array)
		return np.mean(wopt_array)

	def xiw_model_estimate_xi(self,wfinal,gfinal,GT,debug=False):

		def scoring(xivar,wfinal,gfinal,GT):
			score = 0
			nodes = GT[0].nodes()
			for t in range(2,len(GT)+1):
				for i in nodes:
					for j in nodes:
						if i < j:
							if (i,j) in GT[t-1].edges():
								current_edge = 1
							else:
								current_edge = 0
							if (i,j) in GT[t-2].edges():
								previous_edge = 1
							else:
								previous_edge = 0
							edge_copy = current_edge*previous_edge + (1-current_edge)*(1-previous_edge)
							score += np.log(1e-20 + xivar*edge_copy + (1-xivar)*(current_edge*wfinal[gfinal[i]-1,gfinal[j]-1] \
								+ (1-current_edge)*(1 - wfinal[gfinal[i]-1,gfinal[j]-1])))
			return score

		grid_pts = np.linspace(0,1,41)
		current_max = scoring(grid_pts[0],wfinal,gfinal,GT)
		xiopt=0
		score_log = []
		for xivar in grid_pts:
			candidate_score = scoring(xivar,wfinal,gfinal,GT)
			score_log.append(candidate_score)
			if candidate_score >= current_max:
				xiopt = xivar
				current_max = candidate_score
		if debug:
			print 'score log: ',score_log
		return xiopt

	def estimate_params(self,GT,k=2,W=np.eye(2),debug=False):

		flag_estimate_g  = False # False
		flag_estimate_w  = True # False

		ghats = []
		w_hats = {}
		for t in range(len(GT)):
			ghats.append(None)
			w_hats[t] = None
		gfinal = None

		wfinal = None
		xifinal = 0
		time0 = time.time()

		if debug:
			print 'Estimating groups, w, xi. Timing starts here.'


		gtruth = {x[0]:x[1]['group'][0] for x  in GT[0].nodes(data=True)}
		if flag_estimate_g == False:
			gfinal = gtruth
		else:
			#Estimate communities for individual snapshots
			for i,G in enumerate(GT):
				#ghats.append(community.best_partition(G))
				ghats[i] = EstimatorUtility().graph_tool_community(G,k)


			#Aggregate/Unify
			# gfinal = self.unify_communities_sets(ghats,k)
			gfinal = self.unify_communities_LP(ghats,k)

		time1 = time.time()-time0
		if debug:
			for t in range(len(GT)):
				print '\tsnapshot',t,ghats[t]
			print '\tgfinal    ',gfinal
			print '\ttruth     ',gtruth


		if flag_estimate_w==False:
			wfinal = W
		else:
			#Estimate w_hat_t_r_s
			w_hats = {}
			for t,G in enumerate(GT):
				w_hats[t] = np.zeros((k,k))
				for r in range(1,k+1):
					for s in range(1,k+1):
						w_hats[t][r-1,s-1] = self.estimate_w_mle(G,r,s,gfinal) # gtruth # gfinal

			#relate w_hats_t to ws
			wfinal = np.zeros((k,k))
			for r in range(1,k+1):
				for s in range(1,k+1):
					wfinal[r-1,s-1] = self.xiw_model_estimate_w(w_hats,r,s)

		time2 = time.time()- time0
		if debug:
			for t in range(1,len(GT)+1):
				print '\n\t w_hats',t,w_hats[t-1]
			print '\twfinal', wfinal

		#estimate xi exhausively
		if debug:
			print '\tEstimating xi start at time',time2
		xifinal = self.xiw_model_estimate_xi(wfinal,gfinal,GT) # wfinal # W
		time3 = time.time()-time0
		if debug:
			print '\tEstimating xi end at time',time3
			print '\txifinal', xifinal

			
		return ghats,gfinal,w_hats,wfinal,xifinal,[time1,time2,time3]

#Proposed Estimator for the Fixed Group Bernoulli Model 
class EstimatorFixedGroupBernoulli(object):

	def exhaustive_search_no_averaging(w_hats,r,s,debug=False):

		def scoring(xivar,wvar,w_hats,r,s):
			score = 0
			for t in range(1,len(w_hats)+1):
				score += abs(w_hats[t-1][r-1,s-1] - pow(xivar,t-1)*wvar + wvar*pow(1-xivar,t-1))
			return score
		grid_pts = np.linspace(0,1,41)
		current_min = scoring(grid_pts[0],grid_pts[0],w_hats,r,s)
		xiopt,wopt = 0,0
		xvals,yvals,zvals = [],[],[]
		for xivar in grid_pts:
			for wvar in grid_pts:
				candidate_score = scoring(xivar,wvar,w_hats,r,s)
				if candidate_score <= current_min:
					xiopt,wopt = xivar,wvar
					current_min = candidate_score
				if debug:
					xvals.append(xivar)
					yvals.append(wvar)
					zvals.append(candidate_score)

		if debug:
			from matplotlib import pyplot
			from mpl_toolkits.mplot3d import Axes3D
			fig = pyplot.figure()
			ax = Axes3D(fig)
			ax.scatter(xvals,yvals,zvals)
			ax.set_xlabel('xi')
			ax.set_ylabel('w')
			pyplot.show()

		return xiopt,wopt

	def exhaustive_search_with_averaging(w_hats,r,s,debug=False):

		def scoring(xivar,wvar,w_hats,r,s,t):
			return abs(w_hats[t-1][r-1,s-1] - pow(xivar,t-1)*wvar + wvar*pow(1-xivar,t-1))

		xiopt_array,wopt_array = [],[]
		for t in range(1,len(w_hats)+1):

			grid_pts = np.linspace(0,1,41)
			current_min = scoring(grid_pts[0],grid_pts[0],w_hats,r,s,t)
			xiopt,wopt = 0,0
			xvals,yvals,zvals = [],[],[]
			for xivar in grid_pts:
				for wvar in grid_pts:
					candidate_score = scoring(xivar,wvar,w_hats,r,s,t)
					if candidate_score <= current_min:
						xiopt,wopt = xivar,wvar
						current_min = candidate_score
			xiopt_array.append(xiopt)
			wopt_array.append(wopt)

		return np.mean(np.array(xiopt_array)),np.mean(np.array(wopt_array))

#Modified Estimator for Zhang 2016 Model A that Includes Arriving/Departing Nodes
class EstimatorZhangAModified(object):

	def get_statistics(self,GT):

		n_nodes = len(GT[0].nodes())
		t1,t2,t3,t4,t5 = 0,0,0,0,0
		for j in range(n_nodes):
			for i in range(0,j):

				if (i,j) in GT[0].edges():
					t1 += 1

				t2temp = 0
				for t in range(1,len(GT)):
					if (i,j) not in GT[t-1].edges() and (i,j) in GT[t].edges():
						t2temp += 1
				t2 +=t2temp

				t3temp = 0
				for t in range(1,len(GT)):
					if (i,j) not in GT[t-1].edges():
						t3temp += 1
				t3 +=t3temp

				t4temp = 0			
				for t in range(1,len(GT)):
					if (i,j) in GT[t-1].edges() and (i,j) not in GT[t].edges():
						t4temp += 1
				t4 +=t4temp

				t5temp = 0
				for t in range(1,len(GT)):
					if (i,j) in GT[t-1].edges():
						t5temp += 1
				t5 +=t5temp


		meta = {'t1':t1,'t2':t2,'t3':t3,'t4':t4,'t5':t5}

		return meta

	def estimate_random_dynamic_no_arrival_recursive(self,GT):

		meta = self.get_statistics(GT)

		Kmax = 10000
		t1,t2,t3,t4,t5 = meta['t1'],meta['t2'],meta['t3'],meta['t4'],meta['t5']
		n_nodes = len(GT[0].nodes())
		pterm = n_nodes*(n_nodes-1)*.5*0.5 #initialize
		beta = 0.5 #initialize
		for k in range(Kmax):
			alpha = (t1 - pterm +t2)/(t1 - pterm +t3)
			pterm = n_nodes*(n_nodes-1)*.5*(alpha/(alpha+beta))
			beta = (pterm - t1 + t4)/(pterm -t1 + t5)
			pterm = n_nodes*(n_nodes-1)*.5*(alpha/(alpha+beta))

		return alpha,beta

	def estimate_random_dynamic_no_arrival_gridsearch(self,GT):

		meta = self.get_statistics(GT)

		t1,t2,t3,t4,t5 = meta['t1'],meta['t2'],meta['t3'],meta['t4'],meta['t5']
		eps = 1e-4
		n_nodes = len(GT[0].nodes())
		gridvals = np.linspace(eps,1-eps,10)
		L = np.zeros((len(gridvals),len(gridvals)))
		maxidx = [0,0]
		for i,alpha in enumerate(gridvals):
			for j,beta in enumerate(gridvals):
				L[i,j] = round(t1*np.log(alpha/(alpha+beta)) + (n_nodes*(n_nodes-1)*.5 - t1)*np.log(beta/(alpha+beta))	\
					+t2*np.log(alpha) + (t3-t2)*np.log(1-alpha) + t4*np.log(beta) + (t5-t4)*np.log(1-beta),2)

				if L[i,j] > L[maxidx[0],maxidx[1]]:
					maxidx[0],maxidx[1] = i,j

		print np.round(gridvals,2)
		# pprint.pprint(L)
		print 'best alpha',gridvals[maxidx[0]]
		print 'best beta',gridvals[maxidx[1]]

		return gridvals[maxidx[0]],gridvals[maxidx[1]]

	def get_statistics_arrivals(self,GT):

		AT = [nx.adjacency_matrix(G) for G in GT]

		#lambda = (termA -nC2p + termB + termC - sumt(nptC2)p) / ( termD - nC2p + termE + termF - sumt(nptC2)p)  )
		
		termlambda = {x:0 for x in 'ABCDEF'}

		#term A and D
		n0 = len(GT[0].nodes())
		for j in range(n0):
			for i in range(0, j):
				termlambda['A'] += AT[0][i,j]

		termlambda['D'] = termlambda['A']

		#term B and E
		for t in range(1,len(GT)):
			nptminus1 = len(GT[t-1].nodes())
			for j in range(nptminus1):
				for i in range(0, j):
					termlambda['B'] += (1-AT[t-1][i,j])*AT[t][i,j]

					termlambda['E'] += 1-AT[t-1][i,j]

		#terms C and F
		for t in range(1, len(GT)):
			npt = len(GT[t].nodes())
			j = npt-1
			for i in range(0, j):
					termlambda['C'] += AT[t][i,j]

		termlambda['F'] = termlambda['C']



		#mu = (termU -nC2p + termV + termW - sumt(nptC2)p) / ( termX - nC2p + termY + termZ - sumt(nptC2)p)  )

		termmu = {x:0 for x in 'UVWXYZ'}

		#termmu U and X
		n0 = len(GT[0].nodes())
		for j in range(n0):
			for i in range(0, j):
				termmu['U'] += (-AT[0][i,j])

		termmu['X'] = termmu['U']

		#termmu V and Y
		for t in range(1,len(GT)):
			nptminus1 = len(GT[t-1].nodes())
			for j in range(nptminus1):
				for i in range(0, j):
					termmu['V'] += AT[t-1][i,j]*(1 - AT[t][i,j])

					termmu['Y'] += AT[t-1][i,j]

		#termmu W and Z
		for t in range(1, len(GT)):
			npt = len(GT[t].nodes())
			j = npt-1
			for i in range(0, j):
					termmu['W'] += (-AT[t][i,j])

		termmu['Z'] = termmu['W']


		coeff_nC2p = len(GT[0].nodes())*(len(GT[0].nodes()) - 1)*0.5
		coeff_sumt_nptC2p = 0
		for t in range(1, len(GT)):
			npt = len(GT[t].nodes())
			for i in range(0, npt-1):
					coeff_sumt_nptC2p += 1


		return {'termlambda':termlambda,'termmu':termmu,'coeff_p':[coeff_nC2p,coeff_sumt_nptC2p]}

	def estimate_random_dynamic_with_arrival_recursive(self,GT):


		st = time.time()
		print "Estimating parameters"

		meta = self.get_statistics_arrivals(GT)
		# print meta['termlambda']
		# print meta['termmu']
		# print meta['coeff_p']


		p = 0.5 #initialize
		lmbd_prev = 0.5
		mu_prev = 0.5
		# count = 0
		while 1:
			lmbd = (meta['termlambda']['A'] - meta['coeff_p'][0]*p \
					+meta['termlambda']['B'] + meta['termlambda']['C'] - meta['coeff_p'][1]*p)\
					/(meta['termlambda']['D'] - meta['coeff_p'][0]*p \
					+meta['termlambda']['E'] + meta['termlambda']['F'] - meta['coeff_p'][1]*p)

			mu = (meta['termmu']['U'] + meta['coeff_p'][0]*p \
					+meta['termmu']['V'] + meta['termmu']['W'] + meta['coeff_p'][1]*p)\
					/(meta['termmu']['X'] + meta['coeff_p'][0]*p \
					+meta['termmu']['Y'] + meta['termmu']['Z'] + meta['coeff_p'][1]*p)

			# count += 1
			# print 'count',count,'mu',mu
			print 'mu',mu
			
			if abs(lmbd-lmbd_prev)<1e-4 and abs(mu-mu_prev)<1e-4:
				break

			#updates
			p = lmbd/(lmbd+mu)
			lmbd_prev,mu_prev = lmbd,mu


		print "\t Time taken: ",time.time() - st

		return lmbd,mu




if __name__=='__main__':

	n = 10
	negerr = 1
	grps = [1,2,3]
	tau1 = np.array([[0,1,0],[0,0,1],[1,0,0]])
	tau2 = np.array([[0,0,1],[1,0,0],[0,1,0]])
	ghats = {}
	ghats[0] = {}
	for i in range(1,n+1):
		ghats[0][i] = np.random.choice(grps,1)[0]
		# print i,np.argmax(tautrue[ghats[0][i]-1,])
	ghats[1], ghats[2] = {},{}
	for i in range(1,n+1):
		if np.random.rand() < negerr:
			ghats[1][i] = np.argmax(tau1[ghats[0][i]-1,])+1
		else:
			ghats[1][i] = 1
		if np.random.rand() < negerr:
			ghats[2][i] = np.argmax(tau2[ghats[0][i]-1,])+1
		else:
			ghats[2][i] = 2
	gfinal = EstimatorFixedGroupLazy().unify_communities_LP(ghats,len(grps)) #,taus,Qs