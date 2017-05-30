#graph parameter estimation
import numpy as np
import pprint


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
	gridvals = np.linspace(eps,1-eps,6)
	L = np.zeros((len(gridvals),len(gridvals)))
	maxidx = [0,0]
	for i,alpha in enumerate(gridvals):
		for j,beta in enumerate(gridvals):
			L[i,j] = round(t1*np.log(alpha/(alpha+beta)) + (n_nodes*(n_nodes-1)*.5 - t1)*np.log(beta/(alpha+beta))	\
				+t2*np.log(alpha) + (t2-t3)*np.log(1-alpha) + t4*np.log(beta) + (t5-t4)*np.log(1-beta),2)

			if L[i,j] > L[maxidx[0],maxidx[1]]:
				maxidx[0],maxidx[1] = i,j

	print np.round(gridvals,2)
	pprint.pprint(L)
	print 'best alpha',gridvals[maxidx[0]]
	print 'best beta',gridvals[maxidx[1]]

	return gridvals[maxidx[0]],gridvals[maxidx[1]]
