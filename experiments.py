import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
np.random.seed(1000)
from graph_generators import generate_synthetic_dynamic_graphs_fixed_grouping
from graph_estimators import estimate_parameters_dynamic_graphs_fixed_grouping
import time



if __name__=='__main__':
	debug = False
	k = 2
	T = 20
	W = np.array([[.9,.1],[.1,0.9]])#[[1,.0],[.0,1]])# #np.random.rand(k,k)
	GT = generate_synthetic_dynamic_graphs_fixed_grouping(xi=0.5,W=W,n=40,k=k,flag_draw=False,total_time=T)

	ghats,gfinal,w_hats,wfinal,xifinal = estimate_parameters_dynamic_graphs_fixed_grouping(GT,k,W)

	if debug:
	
		G = GT[0]
		partition = ghats[0]
		size = float(len(set(partition.values())))
		pos = nx.spring_layout(G)
		col_vec = np.zeros(len(G.nodes()))
		for com in set(partition.values()) :
			list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
			print 'nodes:',list_nodes
			for k in list_nodes:
				col_vec[k-1] = com
		nx.draw(G, pos, with_labels = True,node_color = list(col_vec))
		plt.show()