import numpy as np
from math import gcd
import networkx as nx
import matplotlib.pyplot as plt


def Beta_from_series_exp(n,Exps):
	"""
	Returns Puiseux characteristic exponents Beta given the multiplicity n and the 
	exponents Exps appearing in the Puiseux series of a good parametrization.
	"""

	Beta = [n]
	e = [n]

	e_curr = n
	while e_curr != 1:
		index = 0
		while (Exps[index]%e_curr == 0):
			index += 1

		beta_curr = Exps[index]
		e_curr = gcd(e_curr, beta_curr)

		Beta.append(beta_curr)
		e.append(e_curr)

	return Beta, e


def blowup_charexp(Beta):
	"""
	Computes a blowup via the Puiseux characteristic exponents.
	Check Thm. 3.5.5 in [Wal04] for details and proof.

	REFERENCES:
	- [Wal04] CTC Wall. Singular points of plane curves. 63. 
			  Cambridge University Press, 2004
	"""

	new_Beta = []
	m = Beta[0]
	b1 = Beta[1]

	if b1 > 2*m:
		new_Beta.append(m)
		for i in range(1,len(Beta)):
			new_Beta.append(Beta[i] - m)
	elif m % (b1 - m) != 0:
		new_Beta.append(b1 - m)
		new_Beta.append(m)
		for i in range(2,len(Beta)):
			new_Beta.append(Beta[i] - Beta[i-1] + m)
	else:
		new_Beta.append(b1 - m)
		for i in range(2,len(Beta)):
			new_Beta.append(Beta[i] - Beta[i-1] + m)
	return new_Beta


def multipl_from_charexp(Beta):
	"""
	Returns the multiplicities of the strict transform Mult from the Puiseux 
	characteristic exponents, that is Beta = (n; beta1, ..., betag)
	"""

	Mult = [Beta[0]]

	while len(Beta) > 1:
		Beta = blowup_charexp(Beta)
		Mult.append(Beta[0])

	# Extra 1?
	Mult.append(1)
	return Mult


def proximity_matrix_from_mult(Mult):
	"""
	Returns the proximity matrix P given the multiplicities of the strict transform Mult.
	"""

	s = len(Mult)

	P = np.eye(s, dtype=int)

	for i in range(0,s-1):
		sum_prox_m = 0

		j = i+1
		while sum_prox_m < Mult[i]:
			sum_prox_m += Mult[j]
			P[j][i] = -1
			j = j+1

	return P


def info_from_charexp(Beta):
	"""
	Computes several information of the plane curve, given the Puiseux characteristic 
	exponents Beta = (n; beta1, ..., betag).

	Returns the Zariski pairs {(q,n)}, the gcd quantities {e}, the reduced Puiseux 
	exponents {m}, the semigroup generators {Semig}, its conductor {conductor} and the 
	reduced quantities {m_bar} = Semig/e.
	"""

	e = [Beta[0]]*len(Beta)
	for i in range(1, len(Beta)):
		e[i] = gcd( e[i-1], Beta[i] )

	n = [0] + [  int(e[i-1]/e[i]) for i in range(1, len(Beta)) ] 

	m = [0] + [ int(Beta[i]/e[i]) for i in range(1, len(Beta)) ]

	q = [0] + [  m[i]-n[i]*m[i-1] for i in range(1, len(Beta)) ]

	Semig = [Beta[0], Beta[1]] 
	for i in range(2, len(q)):
		Semig.append( n[i-1] * Semig[i-1] - Beta[i-1] + Beta[i] )

	conductor = n[-1]*Semig[-1] - Beta[-1] - (Beta[0]-1)

	mbar  = [ int(Beta[1]/e[1]) ] + [ int(Semig[i]/e[i]) for i in range(2, len(q)) ]

	#return list(zip(Q[1:],n[1:]), 
	return q[1:], n[1:], e, m, Semig, conductor, mbar


def autointersection_matrix(P):
	"""
	Given the proximity matrix P, returns the autointersection matrix P^t*P.
	"""
	return np.matmul(P.transpose(),P)


def edge_decoration(A,i,j):
	"""
	Computes edge decoration next to Ei, along edge Ei->Ej, where Ei, Ej are 
	exceptional divisors and A represents the autointersection matrix.
	"""
	
	seen = set()
	# visit j and all neighbors of j (except i) recursively
	to_see = set([j])
	while to_see:
		node = to_see.pop()
		seen.add(node)

		# Get neighbors 
		for n in range(0,len(A)):
			# !ATTENTION: +-1 according to criteria in definition A = Pt*P
			if (n != i and A[node][n] == -1):
				# If not seen, add them to_see
				if n not in seen:
					to_see.add(n)
	
	indices = list(seen)#.sort()
	
	# Select submatrix to compute the determinant
	subA = A[np.ix_(indices,indices)]
	decor = int(abs(np.linalg.det(subA)))

	return decor


def dual_graph(A):
	"""
	Returns the matrix representation G of the dual graph obtained from the 
	information in the autointersection matrix A.
	"""

	n_div_exc = len(A)

	G = nx.from_numpy_matrix(-A)

	return G

def dual_graph_ST(A):
	"""
	Returns the matrix representation G of the dual graph obtained from the information 
	in the autointersection matrix A, and adding the strict transform information. 
	"""

	n_div_exc = len(A)

	G = nx.from_numpy_matrix(-A)

	# Add strict transform through last vertex
	G.add_node(n_div_exc)
	G.add_edge(n_div_exc-1, n_div_exc)

	return G


def plot_decorated_dual_graph(A):
	"""
	Plots the decorated dual graph from the information in the autointersection matrix A. 
	Additionally, each exceptional divisor includes in parenthesis their autointersection 
	number. The strict transform is represented by the label ST, instead of the usual arrow.

	The decorations and their properties are described in [Bla24], and are depicted in red. 
	They are added for each directed egde, and placed closer to the initial edge considered.

	REFERENCES:
	 - [Bla24] G Blanco Fernandez. Topological roots. 
		 		Arxiv preprint: ?. 2024
	"""

	n_div_exc = len(A)
	
	G = nx.from_numpy_matrix(-A)

	# Add strict transform passing through the last vertex
	G.add_node(n_div_exc)
	G.add_edge(n_div_exc-1, n_div_exc)

	pos = nx.spring_layout(G)
	plt.figure()

	vertex_labels = {}
	for node in G.nodes():
		if node != n_div_exc:
			vertex_labels[node] = f"E{node} ({A[node,node]})" 
		else:
			vertex_labels[node] = "ST"

	node_sizes = [250]*(n_div_exc+1)
	node_sizes[-1] = 0

	nx.draw(
	    G, pos, edge_color="black", width=1, linewidths=1,
	    node_size=node_sizes, node_color="cyan", alpha=0.9,
	    labels=vertex_labels, font_size=10
	)

	edge_labels = {}
	for n1, n2 in G.edges:
		if n2 == n_div_exc:
			edge_labels[(n1,n2)] = 1 
		elif n1 != n2:
			edge_labels[(n1,n2)] = edge_decoration(A,n1,n2)
			edge_labels[(n2,n1)] = edge_decoration(A,n2,n1)


	nx.draw_networkx_edge_labels(
	    G, pos, edge_labels=edge_labels, label_pos=0.8, 
	    font_color="red", font_size=10#, font_weight="bold"
	)

	plt.axis("off")
	plt.show()


def splice_diag_from_pairs(n,mbar):
	"""
	Plots the splice diagram with decorations, obtained from the values (n, mbar), 
	derived from the Zariski pairs {(q,n)}
	
	See [Bla24] for details, and how to merge splice diagrams for multiple branches.

	REFERENCES:
	 - [Bla24] G Blanco Fernandez. Topological roots. 
		 		Arxiv preprint: code. 2024
	"""

	g = len(n)
	G = nx.Graph()
	
	# Add nodes and edges
	G.add_node(0)
	for i in range(g):
		G.add_node(2*i+1)
		G.add_node(2*i+2)

		G.add_edge(2*i, 2*i+2)
		G.add_edge(2*i+1, 2*i+2)

	# Add strict transform passing through the last vertex
	G.add_node(2*g+1)
	G.add_edge(2*g, 2*g+1)

	pos = nx.spring_layout(G)
	plt.figure()

	vertex_labels = {}
	vertex_labels[2*g+1] = "ST"
	node_sizes = [100]*(2*g+2)
	node_sizes[-1] = 0

	nx.draw(
	    G, pos, edge_color="black", width=1, linewidths=1,
	    node_size=node_sizes, node_color="red", alpha=1,
	    labels=vertex_labels, font_size=10, font_color="red"
	)

	edge_labels = {}
	for i in range(g):
		edge_labels[(2*i+2, 2*i+1)] = n[i]
		edge_labels[(2*i+2, 2*i)] = mbar[i]
		if (i < g-1):
			edge_labels[(2*i+2, 2*i+4)] = 1
	edge_labels[(2*g, 2*g+1)] = 1

	nx.draw_networkx_edge_labels(
	    G, pos, edge_labels=edge_labels, label_pos=0.8, 
	    font_color="red", font_size=10#, font_weight="bold"
	)

	plt.axis("off")
	plt.show()


def splice_diag_from_dual(A):
	"""
	Plots the splice diagram with decorations, obtained from the information of 
	the dual graph given in the form of the	autointersection matrix A.

	The process consists of constructing the dual graph, computing the decorations, 
	and then removing all edges except for those on rupture vertices or supporting 
	arrowheads. See [EN85, Thm. 20.1] for details and proof of the construction.

	REFERENCES:
	- [EN85] D Eisenbud and WD Neumann. Three-dimensional link theory 
			 and invariants of plane curve singularities. 110. 
			 Princeton University Press, 1985
	"""

	n_nodes = len(A)

	G = nx.from_numpy_matrix(-A)

	# Add strict transform passing through the last vertex
	G.add_node(n_nodes)
	G.add_edge(n_nodes-1, n_nodes)

	vertex_labels = {}
	vertex_labels[n_nodes] = "ST"

	def check_degree(node):
		deg = 0
		for k in range(len(A)):
			if k != node:
				deg -= A[node][k]
		if node == len(A)-1:
			deg += 1
		return deg

	edge_labels = {}
	labeled_nodes = set()
	for n1, n2 in G.edges:
		if n2 == n_nodes:
			edge_labels[(n1,n2)] = 1 
		elif n1 != n2:
			if check_degree(n1) >= 3:
				labeled_nodes.add(n1)
				edge_labels[(n1,n2)] = edge_decoration(A,n1,n2)
			if check_degree(n2) >= 3:
				labeled_nodes.add(n2)
				edge_labels[(n2,n1)] = edge_decoration(A,n2,n1)
	print(edge_labels)
	print(A)

	# Prune
	check_nodes = set(G.nodes())
	while check_nodes:
		j = check_nodes.pop()
		
		# List of neighbors
		neighbors = [n for n in G.neighbors(j) if n != j]
		
		# If degree = 2, eliminate vertex
		# Skip the except divisor intersecting the strict transform
		if len(neighbors) == 2 and j != n_nodes-1:
			n1 = neighbors[0]
			n2 = neighbors[1]

			# Add edge and copy decorations
			G.add_edge(n1,n2)
			if n1 in labeled_nodes:
				edge_labels[(n1,n2)] = edge_labels.pop((n1,j))
			if n2 in labeled_nodes:
				edge_labels[(n2,n1)] = edge_labels.pop((n2,j))
			
			# Remove node
			G.remove_node(j)

			print(f"Deleted vertex {j}, who had neighbors {n1}, {n2}")
			print(A)
			print(edge_labels)
			print(check_nodes)

			# Restart search
			check_nodes = set(G.nodes())


	# Plot
	pos = nx.spring_layout(G)
	plt.figure()

	node_sizes = [100]*len(G.nodes())
	node_sizes[-1] = 0

	nx.draw(
	    G, pos, edge_color="black", width=1, linewidths=1,
	    node_size=node_sizes, node_color="red", alpha=1,
	    labels=vertex_labels, font_size=10, font_color="red"
	)

	nx.draw_networkx_edge_labels(
	    G, pos, edge_labels=edge_labels, label_pos=0.8, 
	    font_color="red", font_size=10#, font_weight="bold"
	)

	plt.axis("off")
	plt.show()



"""
# Example computation from exponents in good parametrization of a branch
n = 4
Exps = [6,7]
Beta, e = Beta_from_series_exp(n,Exps)
"""

# Example computation from Puiseux characteristic exponents of a branch
Beta = [4,6,7]

print(f"Puiseux characteristic exponents: {Beta} \n")


Mult = multipl_from_charexp(Beta)
print(f"Multiplicities of ST: {Mult} \n")


P = proximity_matrix_from_mult(Mult)
print(f"Proximity matrix P: \n{P}\n")

invP = np.linalg.inv(P).astype(int)
print(f"Inverse of proximity matrix P^(-1):\n {invP} \n")


print("Numerical data")
N = np.matmul(Mult, invP.transpose())
print(f"N: {N}")

k = np.matmul(np.ones(len(N), dtype=int), invP.transpose())
# Definicio Guillem
k += np.ones(len(N), dtype=int)
print(f"k: {k} \n")


sigma = [-(ki)/Ni for ki,Ni in zip(k,N)]
sigma_rat = [f"-{ki}/{Ni}" for ki,Ni in zip(k,N)]
print(f"sigma: \n {sigma} \n {sigma_rat} \n")


q, n, e, m, Semig, conductor, mbar = info_from_charexp(Beta)
print(f"Zariski pairs (q,n): \n {list(zip(q,n))} \n")

g = len(q)
print(f"Number of brances in dual graph g: {g} \n")

print(f"Semigroup: {Semig}")
print(f"Conductor: {conductor} \n")


A = autointersection_matrix(P)
print(f"Autointersection matrix P^t*P: \n {A} \n")

G = dual_graph(A)
n_div_exc = len(A)

neighbors = [ [n for n in G.neighbors(j) if n != j] for j in G.nodes ]
print(f"Neighbors dual graph (without ST): \n {neighbors} ")

degrees = [len(n) for n in neighbors]
print(f"Degrees dual graph (without ST): \n {degrees} \n")


epsilon = [ [ (N[j]*sigma[i] + k[j]) for j in neighbors[i] ] for i in G.nodes ]
# Add strict transform
epsilon[-1].append(Mult[n_div_exc-1]*sigma[n_div_exc-1])
print(f"Epsilons (with ST): \n {epsilon} \n")


# Plot of the decorated dual graph
plot_decorated_dual_graph(A)

# Plot of the splice diagram from Zariski pairs
splice_diag_from_pairs(n, mbar)

# Plot of the splice diagram from dual graph info
splice_diag_from_dual(A)