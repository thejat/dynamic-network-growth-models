import networkx as nx
import numpy as np
import time, copy
from collections import Counter

def generate_initial_graph(n,k,W):
	#Graph at time zero
	Goriginal=nx.Graph()
	for i in range(1,n+1):
		Goriginal.add_node(i,group=np.random.choice(range(1,k+1),1)) #fixing groups

	for j in range(1,n+1):
		for i in range(1,j):
			if np.random.rand() <= W[Goriginal.node[i]['group']-1,Goriginal.node[j]['group']-1]:
				Goriginal.add_edge(i,j)
	return Goriginal

def generate_fixed_group(params):
	'''
	Graph at 0 is the single original graph
	Graphs at times t-1,...,total_time are the evolved ones
	'''

	dynamic 	= params['dynamic']
	xi 			= params['xitrue']
	Mu 			= params['Mutrue']
	W 			= params['Wtrue']
	n 			= params['n']
	k 			= params['k']
	total_time 	= params['total_time']
	log_start_time 	= params['start_time']
	flag_adversarial= params['spectral_adversarial']

	#Create the first graph
	st = log_start_time
	# print("\tGenerating GT sequence for the", dynamic, "dynamic")
	Goriginal = generate_initial_graph(n,k,W)
	Wold = copy.deepcopy(W)

	#Create the subsequent total_time number of graphs indexed from 1 to total_time
	GT = [Goriginal]
	for t in range(1,total_time+1): #t = 1,2,...,T
		# print('\t\tGraph at snapshot', t, ' time', time.time() - st)

		if flag_adversarial is True:
			W = 1-W #Adversarial

		Gcurrent = nx.Graph()
		for node in GT[t-1].nodes(data=True):
			Gcurrent.add_node(node[0],group=node[1]['group'])

		if dynamic=='bernoulli':
			for i in Gcurrent.nodes():
				for j in Gcurrent.nodes():
					if i < j:
						if (i, j) in GT[t - 1].edges():
							if np.random.rand() > Mu[Gcurrent.node[i]['group'] - 1, Gcurrent.node[j]['group'] - 1]:
								Gcurrent.add_edge(i, j)
						else:
							if np.random.rand() <= (W[Gcurrent.node[i]['group'] - 1, Gcurrent.node[j]['group'] - 1])*(Mu[Gcurrent.node[i]['group'] - 1, Gcurrent.node[j]['group'] - 1]):
								 Gcurrent.add_edge(i, j)
		elif dynamic=='lazy':
			for i in Gcurrent.nodes():
				for j in Gcurrent.nodes():
					if i < j:
						if np.random.rand() <= xi:
							if (i,j) in GT[t-1].edges():
								Gcurrent.add_edge(i,j)
						else:
							if np.random.rand() <= W[Gcurrent.node[i]['group']-1,Gcurrent.node[j]['group']-1]:
								Gcurrent.add_edge(i,j)
		else:
			raise NotImplementedError

		GT.append(Gcurrent)
	if flag_adversarial is False:
		print('\tTime taken for ',dynamic, 'graph sequence generation:', time.time() - st)
	else:
		print('\tTime taken for ADVERSARIAL ',dynamic, ' dynamic graph sequence generation:', time.time() - st)

	return GT

def add_noise(GT,noise_type='random',noise_level=None):
	if noise_level is None:
		noise_level = 0.8
	GTnoisy = []
	if noise_type=='random':
		for G in GT:
			Gnew = G.copy()
			for e in G.edges():
				if np.random.rand() <= noise_level:
						Gnew.remove_edge(*e)
			GTnoisy.append(Gnew)
	else:
		GTnoisy = GT #TBD

	return GTnoisy

def graph_stats_fixed_group(params,GT):
	nodecounts = []
	edgecounts = []
	for G in GT:
		nodecounts.append(len(G.nodes()))
		edgecounts.append(len(G.edges()))

	gtrue = {x[0]:x[1]['group'][0] for x in GT[0].nodes(data=True)} #only works for fixed group
	community_sizes = Counter([gtrue[i] for i in gtrue])

	return {'gtrue':gtrue, 'nodecounts': nodecounts, 'edgecounts': edgecounts, 'community_sizes': community_sizes}

if __name__=='__main__':
	np.random.seed(1000)

	params 					= {}
	params['n'] 			= 100 # size of the graph
	params['Mutrue'] 		= np.array([[.4,.6],[.6,.4]])# [bernoulli]
	params['Wtrue'] 		= np.array([[.4,.2],[.2,.4]])
	params['k'] 			= params['Wtrue'].shape[0] # number of communities
	params['xitrue'] 		= .2 # [lazy]
	params['start_time'] 	= time.time()
	params['spectral_adversarial'] = False
	params['total_time'] 	=  4 # power of 2, number of additional graph snapshots


	params['dynamic'] 		= 'bernoulli'
	GT = generate_fixed_group(params)

	params['spectral_adversarial'] = True
	GT = generate_fixed_group(params)

	params['dynamic'] 		= 'lazy'	
	GT = generate_fixed_group(params)

	params['spectral_adversarial'] = False
	GT = generate_fixed_group(params)