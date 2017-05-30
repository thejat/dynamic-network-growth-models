import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



def generate_random_dynamic_graphs(alpha=0.5,beta=0.6):

	G=nx.Graph()

	alpha = .5
	beta = .6

	G.add_nodes_from(range(5))
	for i in range(0,5):
		for j in range(0,i):
			if np.random.rand() > alpha/(alpha+beta):
				G.add_edge(i,j)
	nx.draw(G,pos=nx.spring_layout(G))  # networkx draw()
	plt.show()  # pyplot draw()

	GT = [G]
	for t in range(1,3): #t = 1,2
		Gcurrent = GT[t-1].copy()
		current_nodes = sorted(Gcurrent.nodes())

		for j in current_nodes:
			for i in current_nodes:
				if i < j:
					if (i,j) in Gcurrent.edges():
						if np.random.rand() > beta/(alpha+beta):
							Gcurrent.remove_edge(i,j)
					else:
						if np.random.rand() > alpha/(alpha+beta):
							Gcurrent.add_edge(i,j)

		new_node_index = len(current_nodes)+1
		Gcurrent.add_node(new_node_index)
		for i in current_nodes:
			if np.random.rand() > alpha/(alpha+beta):
				Gcurrent.add_edge(i,new_node_index)

		GT.append(Gcurrent)

		nx.draw(Gcurrent,pos=nx.spring_layout(Gcurrent))  # networkx draw()
		plt.show()  # pyplot draw()


		return GT

if __name__=='__main__':
	GT = generate_random_dynamic_graphs()