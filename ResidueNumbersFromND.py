import ZetaFunctionsNewtonND

def backtracking(den, coef, index):
    """
    List of possible coefficients used to compute the blowup ray as a linear combination
    of the cone rays. It lists positive integer vectors whose coordiantes are <= den, 
    and are obtained via a simple backtracking.
    """

    N = len(coef)
    if index == N:
        #print(coef)
        return [coef]

    L = [] 
    L = L + backtracking(den, coef, index + 1)
    i = 1
    while coef[index] + i < den:
        new_coef = coef.copy()
        new_coef[index] += i
        L = L + backtracking(den, new_coef, index + 1) 
        i += 1
    return L


def blowup_ray(R):
    """
    Compute the blowup ray added to subdivide a non regular cone given by the columns of R
    Required: the vectors in R are primitive (gcd of its components = 1)
    """

    N = R.nrows()
    M = R.ncols()
    den = 2
    list_coefs = []
    
    while den < 100000:
        list_coefs = backtracking(den, [0]*N, 0)
        
        for coef in list_coefs[1:]:
            x = [0]*M
            for j in range(M):
                for k in range(N):
                    x[j] += coef[k]*R[k][j]/den
            
            # Check if all entries are integers
            all_integers = True
            for xj in x:
                if floor(xj) != xj:
                    all_integers = False
            # Return x in that case
            if all_integers:
                x_gcd = x[0]
                for i in range(M):
                    x_gcd = gcd(x_gcd, x[i])
                x_red = [xj/x_gcd for xj in x]
                return x_red
        # Try next denominator
        den += 1
        list_coefs = []

    # Error, not found for denominator up to 100000 (increase depth cutoff)
    return -1

def simple_partition(fan, printing=False):
    """
    Compute a simple (regular) partition of the given fan
    """

    P = fan.cone_lattice()
    added_rays = []

    # Ordered list of subcones of maximum dimension
    list_cones =  [ls for i in range(1, fan.dim() + 1) for ls in fan.cones(i)]

    # Make simplicial subdivision
    list_simplicial_cones = []
    for cone in list_cones:
        for c in simplicial_partition(cone):
            list_simplicial_cones.append(c)

    max_dim_cones = [c for c in list_simplicial_cones if c.dim() == fan.dim() ]

    # Set containing codes that are to be subdivided (ie. mult > 1)
    cones_to_divide = set()
    for c in max_dim_cones:
        mult = multiplicity(c)
        if mult > 1:
            cones_to_divide.add((c, mult))

    depth = 1
    while cones_to_divide and depth < 1000:
        c, mult  = cones_to_divide.pop()       
        
        R = c.rays().matrix()
        N = R.nrows()
        
        if printing: print(f"Simplifying cone with mult = {mult}, given by rays \n{R}")
        
        # Construction of new ray x 
        x = blowup_ray(R)
        list_simplicial_cones.append(Cone([x]))
        added_rays.append(tuple(x))
        if printing: print(f"New constructed x = {x}\n")
        
        # Remove non-simple cone and add subdivided cones
        list_simplicial_cones.remove(c)
        if printing: print(f"Removing cone =\n{R}\n")
        for j in range(N):
            new_c_list_rays = [R[i] for i in range(N) if i != j] + [tuple(x)]
            new_c = Cone(new_c_list_rays)

            # Add only if it is of maximum dimension
            if new_c.dim() == fan.dim():
                list_simplicial_cones.append(new_c)   
                if printing: print(f"Adding cone = \n {new_c_list_rays}\n")

                # Check if subcones are simple 
                new_mult = multiplicity(new_c)
                if new_mult > 1:
                    cones_to_divide.add((new_c,new_mult))  
                    if printing: print(f"...also to divide, mult = {new_mult} \n")
                
        depth += 1
    
    return list_simplicial_cones, added_rays


def print_check_simple(max_dim_cones):
    """
    Prints a check that all the maximum dimension cones are simple: checks that the matrix defined 
    by its rays has determinant equal to 1 in absolute value.
    """

    for c in max_dim_cones:
        print("Cone:")
        print(c.rays().matrix())
        print(f"... |det| = {abs(c.rays().matrix().det())}")

def plot_subdivided_fan(max_dim_cones):
    """
    Plots the subdivision of the dual fan, given the maximum dimension cones 
    """

    subdividedFan = Fan(max_dim_cones)
    return subdividedFan.plot()


def is_ray_in_cone(r, c):
    """
    Checks if ray r is one of the generators of cone c
    """

    r = vector(r)
    c_matrix = c.rays().matrix()
    for ray in c_matrix:
        if ray == r:
            return True
    return False

def neighbor_divisors(r, max_dim_cones, printing=False):
    """
    Returns neighbors to ray r, that is, other rays of the fan that appear in a same cone with r
    """

    r = vector(r)
    neighbors = set()
    for cone in max_dim_cones:
        if is_ray_in_cone(r, cone):
            cone_matrix = cone.rays().matrix()
            for ray in cone_matrix:
                if ray != r:
                    if printing: print(f"Adding neighbor ray: {ray}")
                    neighbors.add(tuple(ray))
    return neighbors

def dot_prod(a, b):
    return vector(a)*vector(b)

def get_k_N(r, g):
    """
    Returns numerical data k, N for ray r, in the dual fan of g
    """
    k = sum(r)
    
    list_possible_N = [dot_prod(exp, r)  for exp in g.exponents()]  
    N = min(list_possible_N)

    return k, N

def all_neighbors_k_N(r, max_dim_cones, g):
    """
    Returns numerical data k, N for ray r and all of its neighbors in the dual fan of polynomial g
    """
    neighbors = neighbor_divisors(r, max_dim_cones)

    info_neighbors = []
    for ray in neighbors:
        k, N = get_k_N(ray, g)
        #print(f"Ray {ray} has (k,N) = ({k},{N})")
        info_neighbors.append( [ray, k, N] )
    return info_neighbors


def rays_k_N(g):
    """
    Returns the numerical data of all original rays of the Newton dual fan of the polynomial g
    """
    # Construct Newton polyhedron and dual fan
    P = newton_polyhedron(g)
    F = fan_all_cones(P)
    
    # Save original rays and coordinate rays
    original_rays = [tuple(r) for r in F.rays()]

    # Max dimension cones 
    L = F.cones(F.dim())
    max_dim_cones = [l for l in L]
  
    data_ray = dict()
    for ray in original_rays:
        data_ray[ray] = get_k_N(ray, g)
    return data_ray


def get_epsilons(g, printing = True):
    """
    Computes the residue numbers (epsilons) for each divisor, corresponding to a ray of the 
    orginal dual fan associated to the polynomial g, and its respective neighbor rays in the 
    regular subdivided dual fan.

    Returns:
     - epsilons: a dictionary, for each ray gives another dictionary including
        the neighbor rays and the corresponding residue number
     - possible_val: a sete with the residue numbers found in the example
     - max_dim_cones: the list of maximum dimension cones, obtained after
        the regular subdivision (required for the subdivided fan plot)
     - original_rays: list of the original rays of the dual fan
     - added_rays: list of the added rays in the regular subdivision
    """

    zex = ZetaFunctions(g)
    zex.topological_zeta(local = True)
    s = var("s")
    
    # Construct Newton polyhedron and dual fan
    P = newton_polyhedron(g)
    F = fan_all_cones(P)
    
    # Save original rays and coordinate rays
    original_rays = [tuple(r) for r in F.rays()]
    Id = matrix.identity(n_vars)
    coordinate_rays = set( [tuple(Id[j]) for j in range(n_vars)] )
    
    # Simplicial regular (simple) subdivision
    L, added_rays = simple_partition(F, printing = False)
    # Max dimension cones of the subdivision
    max_dim_cones = [l for l in L if l.dim() == F.dim()]
    
    chart_rays = list(set(original_rays).difference(coordinate_rays))
    total_rays = original_rays + added_rays
    
    epsilons = dict()
    possible_val = set() 
    
    for ray in chart_rays:
        k, N = get_k_N(ray, g)
        
        eps_ray = dict()
        eps_ray["ST"] = -k/N
        possible_val.add( -k/N )
    
        sum_eps = -k/N
            
        info_neighbors = all_neighbors_k_N(ray, max_dim_cones, g)
        #print(info_neighbors)

        for elem in info_neighbors:
            neigh = elem[0]
            k_neigh = elem[1]
            N_neigh = elem[2]
            eps = - N_neigh * k/N + k_neigh
            eps_ray[neigh] = eps
            possible_val.add(eps)
            sum_eps += eps

            ## debugging
            if eps > 1 and eps == round(eps):
                print(f"{g} HAS INTEGER EPSILON > 1: {eps}")
        
        epsilons[ray] = [eps_ray, sum_eps]

    if printing:
        print("")
        for key in epsilons:
            print(f"{key} has epsilons: {epsilons[key][0]} \n")
    
    print(f"\n Possible epsilons found: {possible_val}")

    return epsilons, possible_val, max_dim_cones, original_rays, added_rays




def gcd_menors(A, B):
    """
    Computes the gcd of the 2x2 minors of the matrix formed by taking vectors A,B as columns.
    This is the beta factor defined by Loeser in [p. 87, Loe90].

    REFERENCES
     - [Loe90] F. Loeser. "Fonctions d'Igusa p-adiques, polynomes de Bernstein, 
                et polyedres de Newton" (1990). 
    """

    n = len(A)
    pairs = [ [i,j] for i in range(n) for j in range(i+1,n) ]
    list_menors = [ abs(A[i]*B[j]-A[j]*B[i]) for i,j in pairs  ]

    return gcd(list_menors)


def get_loeser_epsilons(g, printing=True):
    """
    Computes the Loeser residue numbers (epsilons) for each divisor, corresponding to a ray 
    of the orginal dual fan associated to the polynomial g, and its respective neighbor rays 
    in the original dual fan (that is without regular subdvision!)

    Returns:
     - epsilons: a dictionary, for each ray gives another dictionary including
        the neighbor rays and the corresponding residue number
     - possible_val: a sete with the residue numbers found in the example
    """
    
    # Construct Newton polyhedron and dual fan
    P = newton_polyhedron(g)
    F = fan_all_cones(P)
    
    # Save original rays and coordinate rays
    original_rays = [tuple(r) for r in F.rays()]
    
    Id = matrix.identity(n_vars)
    coordinate_rays = set( [tuple(Id[j]) for j in range(n_vars)] )
    chart_rays = list(set(original_rays).difference(coordinate_rays))

    # Max dimension cones 
    L = F.cones(F.dim())
    max_dim_cones = [l for l in L]
  
    epsilons = dict()
    possible_val = set() 
    
    for ray in chart_rays:
        k, N = get_k_N(ray, g)
        
        eps_ray = dict()
        eps_ray["ST"] = -k/N
        possible_val.add( -k/N )
    
        sum_eps = -k/N
            
        info_neighbors = all_neighbors_k_N(ray, max_dim_cones, g)
        #print(info_neighbors)

        for elem in info_neighbors:
            neigh = elem[0]
            k_neigh = elem[1]
            N_neigh = elem[2]
            eps = - N_neigh * k/N + k_neigh

            # Loeser factor beta
            beta = gcd_menors(ray, neigh)
            eps /= beta            
            
            eps_ray[neigh] = eps
            possible_val.add(eps)
            sum_eps += eps

            ## debugging
            if eps > 1 and eps == round(eps):
                print(f"{g} HAS INTEGER EPSILON > 1: {eps}")
        
        epsilons[ray] = [eps_ray, sum_eps]

    if printing:
        print("")
        for key in epsilons:
            print(f"{key} has Loeser epsilons: {epsilons[key][0]} \n")
    
    print(f"\n Possible epsilons found: {possible_val}")

    return epsilons, possible_val






## Outdated functions
###########################################################################
"""

## Numerical data, auxiliary functions

def all_k_N(rays, g):
    
    #Returns numerical data k, N of all the desired rays
    
    info_all = []
    for ray in rays:
        k, N = get_k_N(ray, g)
        #print(f"Ray {ray} has (k,N) = ({k},{N})")
        info_all.append( [ray, k, N] )
    return info_all


## Exhaustive search

# hiperplane given by <H,x> = c
def backtrack_exponents(H, c, coords, index, current_c):
    n = len(H)
    if index == n:     
        return [coords]

    L = []
    L = L + backtrack_exponents(H, c, coords, index+1, current_c)

    i = 1
    while current_c + i*H[index] <= c:
        new_coords = coords.copy()
        new_coords[index] += i
        L = L + backtrack_exponents(H, c, new_coords, index+1, current_c+i*H[index])
        i += 1
    return L

def get_exponents_below_hip(H, c):
    coords = [0]*len(H)
    L = backtrack_exponents(H, c, coords, 0, 0)
    L.remove(coords)
    return L

def num_to_rev_binarystring(n):
    if n <= 1:
        return str(n)
    return str(n%2) + num_to_rev_binarystring(n//2)

def select_subset(L, bin_str):
    S = []
    index = 0
    for c in bin_str:
        if c == "1":
            S.append(L[index])
        index += 1
    return S

def get_subsets_list(L):
    PL = []

    n = len(L)
    for i in range(2**n):
        bin_str = num_to_rev_binarystring(i)
        #print(f"{i} corresp to {bin_str} then: {select_subset(L, bin_str)}")
        PL.append( select_subset(L, bin_str) )
    return PL

def get_poly_as_subset_monom(Monomials, bin_str):
    f = 0
    index = 0
    for c in bin_str:
        if c == "1":
            f += Monomials[index]
        index += 1
    return f

def get_polys_list(Monomials):
    L = []

    n = len(Monomials)
    for i in range(1, 2**n):
        bin_str = num_to_rev_binarystring(i)
        #print(f"{i} corresp to {bin_str} then: {select_subset(L, bin_str)}")
        L.append( get_poly_as_subset_monom(Monomials, bin_str) )

    return L



## Simplified versions for exhaustive search

# Simplified/sped up version used during exhaustive search for bad 
# residue numbers with regular subdivision
def get_epsilons_simplified(g, printing = True):
    #zex = ZetaFunctions(g)
    #zex.topological_zeta(local = True)
    #s = var("s")
    
    # Construct Newton polyhedron and dual fan
    P = newton_polyhedron(g)
    F = fan_all_cones(P)
    
    # Save original rays and coordinate rays
    original_rays = [tuple(r) for r in F.rays()]
    Id = matrix.identity(n_vars)
    coordinate_rays = set( [tuple(Id[j]) for j in range(n_vars)] )
    
    
    # Simplicial regular (simple) subdivision
    L, added_rays = simple_partition(F)
    # Max dimension cones of the subdivision
    max_dim_cones = [l for l in L if l.dim() == F.dim()]
    
    chart_rays = list(set(original_rays).difference(coordinate_rays))
    
    greater_than_one = False
    max_eps = -10
    
    for ray in chart_rays:
        k, N = get_k_N(ray, g)
        
        info_neighbors = all_neighbors_k_N(ray, max_dim_cones, g)
        
        for elem in info_neighbors:
            neigh = elem[0]
            k_neigh = elem[1]
            N_neigh = elem[2]
            eps = - N_neigh * k/N + k_neigh
            if eps > 1:
                greater_than_one = True
                max_eps = max(max_eps, eps)
    
    if greater_than_one:
        print(f"{g} has epsilon > 1: {max_eps} \n")
    return


# Simplified/sped up version used during exhaustive search for bad 
# Loeser residue numbers
def get_loeser_simplified_epsilons(g, printing=True):
    
    # Construct Newton polyhedron and dual fan
    P = newton_polyhedron(g)
    F = fan_all_cones(P)
    
    # Save original rays and coordinate rays
    original_rays = [tuple(r) for r in F.rays()]
    
    Id = matrix.identity(n_vars)
    coordinate_rays = set( [tuple(Id[j]) for j in range(n_vars)] )
    chart_rays = list(set(original_rays).difference(coordinate_rays))
    
    # Max dimension cones 
    L = F.cones(F.dim())
    max_dim_cones = [l for l in L]

    for ray in chart_rays:
        k, N = get_k_N(ray, g)
        
        info_neighbors = all_neighbors_k_N(ray, max_dim_cones, g)
        #print(info_neighbors)

        for elem in info_neighbors:
            neigh = elem[0]
            k_neigh = elem[1]
            N_neigh = elem[2]
            eps = - N_neigh * k/N + k_neigh

            # Loeser factor beta (pag 87 article 1990)
            beta = gcd_menors(ray, neigh)
            eps /= beta            
            

            if eps > 1 and eps == int(eps):
                print(f"{g} HAS INTEGER EPSILON > 1: {eps}")
        
    return


# Simplified/sped up version used during exhaustive search for bad 
# Loeser residue numbers, checking only compact faces
def get_loeser_simplified_epsilons_compact(g, printing=True):
    
    # Construct Newton polyhedron and dual fan
    P = newton_polyhedron(g)
    F = fan_all_cones(P)
    
    # Save original rays and coordinate rays
    original_rays = [tuple(r) for r in F.rays()]
    
    Id = matrix.identity(n_vars)
    coordinate_rays = set( [tuple(Id[j]) for j in range(n_vars)] )
    chart_rays = list(set(original_rays).difference(coordinate_rays))
    
    # Max dimension cones 
    L = F.cones(F.dim())
    max_dim_cones = [l for l in L]

    for ray in chart_rays:
        k, N = get_k_N(ray, g)
        
        info_neighbors = all_neighbors_k_N(ray, max_dim_cones, g)
        #print(info_neighbors)

        # Check if neighboring coordiante rays => non compact face as neighbor
        coord_neigh = False
        list_neigh = set([ n[0] for n in info_neighbors])
        for c_ray in coordinate_rays:
            if not coord_neigh: #still not found any 
                if c_ray in list_neigh:
                    coord_neigh = True

        if not coord_neigh:
            for elem in info_neighbors:
                neigh = elem[0]
                k_neigh = elem[1]
                N_neigh = elem[2]
                eps = - N_neigh * k/N + k_neigh

                # Loeser factor beta (pag 87 article 1990)
                beta = gcd_menors(ray, neigh)
                eps /= beta            
                

                if eps > 1 and eps == int(eps):
                    print(f"{g} HAS INTEGER EPSILON > 1: {eps}")
        #else:
        #    print(f"{ray} has coord neighbor :(")       

    return
"""