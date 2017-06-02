import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
np.random.seed(1000)
from graph_estimators import estimate_random_dynamic_no_arrival_recursive,estimate_random_dynamic_with_arrival_recursive
import time



def generate_random_dynamic_graphs(alpha=0.5,beta=0.6,n0=50,flag_draw=False,total_time=10,flag_arrivals=True):

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

if __name__=='__main__':

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
