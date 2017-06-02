#graph parameter estimation
import numpy as np
import pprint, time
import networkx as nx


def get_statistics(GT):

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


def estimate_random_dynamic_no_arrival_recursive(GT):

	meta = get_statistics(GT)

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


def estimate_random_dynamic_no_arrival_gridsearch(GT):

	meta = get_statistics(GT)

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


def get_statistics_arrivals(GT):

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
		for j in range(npt):
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
		for j in range(npt):
			for i in range(0, j):
				termmu['W'] += (-AT[t][i,j])

	termmu['Z'] = termmu['W']


	coeff_nC2p = len(GT[0].nodes())*(len(GT[0].nodes()) - 1)*0.5
	coeff_sumt_nptC2p = 0
	for t in range(1, len(GT)):
		npt = len(GT[t].nodes())
		for j in range(npt):
			for i in range(0, j):
				coeff_sumt_nptC2p += 1


	return {'termlambda':termlambda,'termmu':termmu,'coeff_p':[coeff_nC2p,coeff_sumt_nptC2p]}


def estimate_random_dynamic_with_arrival_recursive(GT):


	st = time.time()
	print "Estimating parameters"

	meta = get_statistics_arrivals(GT)
	print meta['termlambda']
	print meta['termmu']
	print meta['coeff_p']

	Kmax = 1

	p = 0.5 #initialize
	for k in range(Kmax):
		lmbd = (meta['termlambda']['A'] - meta['coeff_p'][0]*p \
				+meta['termlambda']['B'] + meta['termlambda']['C'] - meta['coeff_p'][1]*p)\
				/(meta['termlambda']['D'] - meta['coeff_p'][0]*p \
				+meta['termlambda']['E'] + meta['termlambda']['F'] - meta['coeff_p'][1]*p)

		mu = (meta['termmu']['U'] - meta['coeff_p'][0]*p \
				+meta['termmu']['V'] + meta['termmu']['W'] - meta['coeff_p'][1]*p)\
				/(meta['termmu']['X'] - meta['coeff_p'][0]*p \
				+meta['termmu']['Y'] + meta['termmu']['Z'] - meta['coeff_p'][1]*p)

		print 'mu',mu
		

		p = lmbd/(lmbd+mu)

	print "\t Time taken: ",time.time() - st

	return lmbd,mu