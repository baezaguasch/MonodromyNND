"""
This code adds routines to the class
    class ZetaFunctions():
in the code "ZetaFunctionsNewtonND.py" in 
    https://jviusos.github.io/ 
by Juan Viu-Sos (juan.viusos@upm.es), from Universidad Politecnica de Madrid                                                                                            
"""

import ZetaFunctionsNewtonND

def top_zeta_contrib(self, d=1, local=False, weights=None, info=False, check="ideals", print_result=True):
    """
    Return the expression of the topological zeta function and the individual contributions: 
    for each ray, runs through all cone containing it and adds up only these terms.
    The output is a dictionary where the key is the face and the value the contribution.

    INPUTS and WARNING are the same as the function ''topological_zeta''

    EXAMPLES:

        sage: R.<x,y,z> = QQ[]
        sage: zex = ZetaFunctions(x^5 + y^6 + z^7 + x^2*y*z + x*y*z^2 + x*y^2*z)
        sage: s = var("s")
        sage: contrib = zex.top_zeta_contrib(d=1, local=True, info=False)

        > Analyzing contribution of ray (1, 2, 1)
        > Which appears in face given by rays 
        >       [(3, 1, 1), (23, 7, 6), (1, 1, 1), (1, 2, 1), (7, 18, 5)]
        > and contributes by: 
        >   (464/11025) * (s + 6/7)^-2 * (s + 3/4)^-1 * (s + 4/5)^-1 * 
                        * (s + 5/6)^-1 * (s^2 + 12163/7424*s + 1245/1856)
        > ...

    """
    f = self._f
    s = polygen(QQ, "s")
    ring_s = s.parent()
    P = self._Gammaf
    if check != "no_check":
        if local:
            if is_newton_degenerated(f, P, local=True, method=check, info=info):
                raise TypeError("degenerated wrt Newton")
        else:
            if is_global_degenerated(f, method=check):
                raise TypeError("degenerated wrt Newton")
    else:
        print("Warning: not checking the non-degeneracy condition!")
    result = ring_s.zero()
    if local:
        faces_set = compact_faces(P)
    else:
        faces_set = proper_faces(P)
        if d == 1:
            total_face = faces(P)[-1]
            dim_gamma = total_face.dim()
            vol_gamma = face_volume(f, total_face)
            result = (s / (s + 1)) * (-1) ** dim_gamma * vol_gamma
            if info:
                print("Gamma: total polyhedron")
                print("J_gamma = 1")
                print("dim_Gamma!*Vol(Gamma) = " + str(vol_gamma))
                print()

    faces_set = face_divisors(d, faces_set, P)

    dict_rays = dict()
    possible_rays = set()

    for tau in faces_set:
        dict_rays[tau] = [tuple(ray) for ray in cone_from_face(tau).rays()]
        if info:
            print(f"Face info: {face_info_output(tau)}")
            print(f"With rays: {dict_rays[tau]} \n")
        for ray in dict_rays[tau]:
            possible_rays.add(ray)

    if info:
        print(f"Possible rays: {possible_rays} \n")

    contrib = dict()
    
    for ray in possible_rays:
        if print_result:
            print(f"\nAnalyzing contribution of ray {ray}")
        result = 0
        for tau in faces_set:
            if ray in dict_rays[tau]:
                if print_result:
                    rays_face = [tuple(ray) for ray in cone_from_face(tau).rays()]
                    print(f"Which appears in face given by rays {rays_face}")
                J_tau, cone_info = Jtau(tau, P, weights, s)
                dim_tau = tau.dim()
                vol_tau = face_volume(f, tau)

                if d == 1:
                    if dim_tau == 0:
                        term = J_tau
                    else:
                        term = (s / (s + 1)) * ((-1) ** dim_tau) * vol_tau * J_tau
                else:
                    term = ((-1) ** dim_tau) * vol_tau * J_tau
                result += term
                result = simplify(expand(result))
                if result != 0:
                    result = result.factor()
                if print_result:
                    print(f"and contributes by: {result}\n")
        contrib[ray] = result

    return contrib


def top_zeta_contrib_containing_rays(self, cont_rays, d=1, local=False, weights=None, info=False, check="ideals", print_result=True):
    """
    Return the contribution to the topological zeta function only of the terms from faces whose 
    associated dual cone contains the given ray(s) given in {cont_rays}.
    The output is a dictionary where the key is the face and the value the contribution.

    WARNING and INPUTS are the same as the function ''topological_zeta'', except for the additional:
    
    - ''cont_rays'' -- array of rays (each as a tuple) for which to compute the contribution 

    EXAMPLES:

        sage: R.<x,y,z> = QQ[]
        sage: zex = ZetaFunctions(x^5 + y^6 + z^7 + x^2*y*z + x*y*z^2 + x*y^2*z)
        sage: s = var("s")
        sage: cont_rays = [(1,1,2), (1,2,1)]
        sage: contrib_selected = zex.top_zeta_contrib_containing_rays(cont_rays, 
                d=1, local=True, info=True)

        > Analyzing contribution of rays [(1, 1, 2), (1, 2, 1)] 

        > Which appear(s) in face given by rays [(1, 1, 1), (1, 2, 1), (1, 1, 2)]
        > and contributes by: (1/100) * (s + 4/5)^-2 * (s + 3/4)^-1
        >  ...

    """

    f = self._f
    s = polygen(QQ, "s")
    ring_s = s.parent()
    P = self._Gammaf
    if check != "no_check":
        if local:
            if is_newton_degenerated(f, P, local=True, method=check, info=info):
                raise TypeError("degenerated wrt Newton")
        else:
            if is_global_degenerated(f, method=check):
                raise TypeError("degenerated wrt Newton")
    else:
        print("Warning: not checking the non-degeneracy condition!")
    result = ring_s.zero()
    if local:
        faces_set = compact_faces(P)
    else:
        faces_set = proper_faces(P)
        if d == 1:
            total_face = faces(P)[-1]
            dim_gamma = total_face.dim()
            vol_gamma = face_volume(f, total_face)
            result = (s / (s + 1)) * (-1) ** dim_gamma * vol_gamma
            if info:
                print("Gamma: total polyhedron")
                print("J_gamma = 1")
                print("dim_Gamma!*Vol(Gamma) = " + str(vol_gamma))
                print()

    faces_set = face_divisors(d, faces_set, P)

    dict_rays = dict()

    for tau in faces_set:
        dict_rays[tau] = [tuple(ray) for ray in cone_from_face(tau).rays()]
    
    if print_result:
        print(f"\n       Analyzing contribution of rays {cont_rays} \n")

    dict_contrib = dict()

    for tau in faces_set:
        result = 0

        check_face = True
        for ray in cont_rays:
            if not (ray in dict_rays[tau]):
                check_face = False

        if check_face:
            J_tau, cone_info = Jtau(tau, P, weights, s)
            dim_tau = tau.dim()
            vol_tau = face_volume(f, tau)

            if d == 1:
                if dim_tau == 0:
                    term = J_tau
                else:
                    term = (s / (s + 1)) * ((-1) ** dim_tau) * vol_tau * J_tau
            else:
                term = ((-1) ** dim_tau) * vol_tau * J_tau
            result += term
            result = simplify(expand(result))
            if result != 0:
                result = result.factor()

            dict_contrib[tau] = result

            associated_rays = [tuple(ray) for ray in cone_from_face(tau).rays()]
            
            if print_result:
                print(f"Which appear(s) in face given by rays {associated_rays}")
                print(f"and contributes by: \n {result}\n")
    
    return dict_contrib


ZetaFunctions.top_zeta_contrib = top_zeta_contrib
ZetaFunctions.top_zeta_contrib_containing_rays = top_zeta_contrib_containing_rays
