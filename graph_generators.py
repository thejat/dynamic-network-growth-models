import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
np.random.seed(1000)
from graph_estimators import estimate_random_dynamic_no_arrival_recursive,estimate_random_dynamic_with_arrival_recursive, estimate_parameters_dynamic_graphs_fixed_grouping
import time


def generate_synthetic_dynamic_graphs_fixed_grouping(xi=0.5,W=np.eye(2),n=10,k=2,flag_draw=False,total_time=2):

	st = time.time()
	print "Generating data"

	#Graph at time zero
	Goriginal=nx.Graph()
	for i in range(1,n+1):
		Goriginal.add_node(i,group=np.random.choice(range(1,k+1),1)) #fixing groups
	
	for j in range(1,n+1):
		for i in range(1,j):
			if np.random.rand() <= W[Goriginal.node[i]['group']-1,Goriginal.node[j]['group']-1]:
				Goriginal.add_edge(i,j)

	if flag_draw==True:
		print 'node color',[x[1]['group'][0]*1.0/k for x  in Goriginal.nodes(data=True)]
		nx.draw(Goriginal,pos=nx.spring_layout(Goriginal),with_labels = True,node_color=[x[1]['group'][0]*1.0/k for x  in Goriginal.nodes(data=True)])  # networkx draw()
		plt.show()  # pyplot draw()


	#Subsequent graphs
	GT = [Goriginal]
	for t in range(1,total_time+1): #t = 1,2,...,T

		print "\tGraph snapshot at snapshot",t, " time",time.time()-st

		Gcurrent = GT[t-1].copy()
		current_nodes = sorted(Gcurrent.nodes())

		#remove all edges in Gcurrent
		for j in current_nodes:
			for i in current_nodes:
				if i < j:
					if (i,j) in Gcurrent.edges():
						Gcurrent.remove_edge(i,j)

					if np.random.rand() <= xi:
						if (i,j) in GT[t-1].edges():
							Gcurrent.add_edge(i,j)
					else:
						if np.random.rand() <= W[Gcurrent.node[i]['group']-1,Gcurrent.node[j]['group']-1]:
							Goriginal.add_edge(i,j)

		GT.append(Gcurrent)

		if flag_draw==True:
			nx.draw(Gcurrent,pos=nx.spring_layout(Gcurrent),with_labels = True,node_color=[x[1]['group']*1.0/k for x  in Gcurrent.nodes(data=True)])  # networkx draw()
			plt.show()  # pyplot draw()

	print "\tTime taken:",time.time() - st

	return GT

def generate_random_dynamic_graph_zhang_modelA(alpha=0.5,beta=0.6,n0=50,flag_draw=False,total_time=10,flag_arrivals=True):

	st = time.time()
	print "Generating data"

	#Graph at time zero
	Goriginal=nx.Graph()
	Goriginal.add_nodes_from(range(n0))
	for j in range(n0):
		for i in range(0,j):
			if np.random.rand() <= alpha/(alpha+beta):
				Goriginal.add_edge(i,j)
	if flag_draw==True:
		nx.draw(Goriginal,pos=nx.spring_layout(Goriginal))  # networkx draw()
		plt.show()  # pyplot draw()


	#Subsequent graphs
	GT = [Goriginal]
	for t in range(1,total_time+1): #t = 1,2,...,t

		print "\tGraph snapshot at snapshot",t, " time",time.time()-st

		Gcurrent = GT[t-1].copy()
		current_nodes = sorted(Gcurrent.nodes())

		#additions/deletions with existing nodes
		for j in current_nodes:
			for i in current_nodes:
				if i < j:
					if (i,j) in Gcurrent.edges():
						if np.random.rand() <= beta/(alpha+beta):
							Gcurrent.remove_edge(i,j)
					else:
						if np.random.rand() <= alpha/(alpha+beta):
							Gcurrent.add_edge(i,j)

		if flag_arrivals==True:
			#new node arrives #tbd: generalize to Z_t nodes arriving
			new_node_index = len(current_nodes)+1
			Gcurrent.add_node(new_node_index)
			for i in current_nodes:
				if np.random.rand() <= alpha/(alpha+beta):
					Gcurrent.add_edge(i,new_node_index)

		GT.append(Gcurrent)

		if flag_draw==True:
			nx.draw(Gcurrent,pos=nx.spring_layout(Gcurrent))  # networkx draw()
			plt.show()  # pyplot draw()

	print "\tTime taken:",time.time() - st

	return GT

def illustrate_zhang_modelA():

	# alphaTrue = 0.7
	# betaTrue = 0.4

	# st = time.time()
	# GT = generate_random_dynamic_graphs(alpha=alphaTrue,beta=betaTrue,n0=20,flag_draw=False,total_time=1000,flag_arrivals=False)
	# print "generated data, time taken:",time.time() - st

	# st = time.time() 	
	# alpha,beta = estimate_random_dynamic_no_arrival_recursive(GT)
	# print "estimated from data, time taken: ",time.time() - st
	# print "True vals: alpha",alphaTrue," beta",betaTrue
	# print "Estimates: alpha",alpha," beta ",beta
	
	# st = time.time()
	# alpha,beta = estimate_random_dynamic_no_arrival_gridsearch(GT)
	# print "estimated from data, time taken: ",time.time() - st
	# print "True vals: alpha",alphaTrue," beta",betaTrue
	# print "Estimates: alpha",alpha," beta ",beta



	lmbdTrue = 0.7
	muTrue = 0.4

	GT = generate_random_dynamic_graphs(alpha=lmbdTrue,beta=muTrue,n0=20,flag_draw=False,total_time=10,flag_arrivals=True)
	
	lmbd,mu = estimate_random_dynamic_with_arrival_recursive(GT)

	print "True vals: lambda",lmbdTrue," mu",muTrue
	print "Estimates: lambda",lmbd," mu ",mu


if __name__=='__main__':
	debug = False
	k = 2
	W = np.array([[.9,.1],[.1,0.9]])#np.random.rand(k,k)
	GT = generate_synthetic_dynamic_graphs_fixed_grouping(xi=1,W=W,n=60,k=k,flag_draw=False,total_time=2)

	ghats,gfinal,w_hats,wfinal,xifinal = estimate_parameters_dynamic_graphs_fixed_grouping(GT,k)

	if debug:
		print 'truth    ',{x[0]:x[1]['group'][0] for x  in GT[0].nodes(data=True)}	
		print 'snapshot1',ghats[0]
		print 'snapshot2',ghats[1]
		print 'snapshot3',ghats[2]
		print 'final    ',gfinal
	print '\n w_hats',w_hats[0]
	print 'wfinal', wfinal
	print 'xifinal', xifinal

	# G = GT[0]
	# partition = ghats[0]
	# size = float(len(set(partition.values())))
	# pos = nx.spring_layout(G)
	# col_vec = np.zeros(len(G.nodes()))
	# for com in set(partition.values()) :
	# 	list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
	# 	print 'nodes:',list_nodes
	# 	for k in list_nodes:
	# 		col_vec[k-1] = com
	# nx.draw(G, pos, with_labels = True,node_color = list(col_vec))
	# plt.show()