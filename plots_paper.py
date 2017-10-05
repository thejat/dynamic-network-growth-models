import numpy as np
import pickle,pprint

if __name__=='__main__':

	rawdata = pickle.load(open('explog.pkl','rb'))
	log= rawdata['log']
	params = rawdata['params']

	ts_meanw = [np.zeros((params['k'],params['k'])) for t in range(params['total_time']-1)]
	for t in range(params['total_time']-1):
		for mcrun in range(params['n_mcruns']):
			ts_meanw[t] += log[mcrun]['wfinals'][t]

	for t in range(params['total_time']-1):
		ts_meanw[t] = ts_meanw[t]*1.0/params['n_mcruns']

	pprint.pprint(ts_meanw)

	debug= False
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