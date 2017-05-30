#graph parameter estimation

def estimate_random_dynamic_no_arrival(GT):

	Kmax = 1000
	p = 0.5


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



	pterm = n_nodes*(n_nodes-1)*.5*p
	for k in range(Kmax):
		alpha = (t1 - pterm +t3)/(t1 - pterm +t3)
		pterm = n_nodes*(n_nodes-1)*.5*(alpha/(alpha+beta))
		beta = (pterm - t1 + t4)/(pterm -t1 + t5)
		pterm = n_nodes*(n_nodes-1)*.5*(alpha/(alpha+beta))



	return alpha,beta

# estimate_random_dynamic_no_arrival(GT)