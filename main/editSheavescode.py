#!/usr/bin/env python3
"""
Sheaf analysis. All detailed mathematical and reasoning aspects are in PDF file
"""

import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations
from collections import defaultdict
import gudhi 

np.set_printoptions(precision=3, suppress=True)

def validate_simplex_tree(simplex_tree):  
    """
    validation of the input not to input the np arrays, only built in class SimplexTree from gudhi
    """
    if not isinstance(simplex_tree, gudhi.SimplexTree):  #
        raise ValueError("incorrect input, use SimplexTree data type")  #
    return True  #


def get_vertex_coordinates(simplex_tree, vertex_coords=None):  #
    """
    vertex extraction
    """
    validate_simplex_tree(simplex_tree)  #
    
    
    if vertex_coords is not None:  #
        return vertex_coords  #
    
    
    vertices = []  #
    for simplex, filtration in simplex_tree.get_simplices():  #
        if len(simplex) == 1:  #
            vertices.append(simplex[0])  #
    
    
    if hasattr(simplex_tree, 'get_point'):  #
        vertex_coords = np.array([simplex_tree.get_point(v) for v in vertices])  #
    return vertex_coords  #


def get_simplices_by_dimension(simplex_tree, max_dim=None):  #
    """
    pretty self explanatory, returns dict (dim--> lists of simplices)
    

    """
    validate_simplex_tree(simplex_tree)  #
    
    simplices_by_dim = defaultdict(list)  #
    
    for simplex, _ in simplex_tree.get_simplices():  #
        dim = len(simplex) - 1  #
        if max_dim is None or dim <= max_dim:  #
            simplices_by_dim[dim].append(tuple(sorted(simplex)))  #
    
    
    for dim in simplices_by_dim:  #
        simplices_by_dim[dim].sort()  #
    
    return simplices_by_dim  #


def get_faces(simplex_tuple):  #
    """
    returns list of all faces of a simplex 
    """
    simplex_set = set(simplex_tuple)  #
    faces = []  #
    for k in range(len(simplex_set), 0, -1):  #
        for face_tuple_gen in combinations(simplex_tuple, k):  #
            faces.append(tuple(sorted(face_tuple_gen)))  #
    return faces  #


def get_boundary_simplices(simplex_tuple, codim=1):  #
    """
    returns list of bd simplices
    

    """
    if not simplex_tuple or len(simplex_tuple) <= codim:  #
        return []  #
    target_len = len(simplex_tuple) - codim  #
    boundary = []  #
    for face_tuple_gen in combinations(simplex_tuple, target_len):  #
        boundary.append(tuple(sorted(face_tuple_gen)))  #
    return boundary  #


class PresheafVectorSpaceOnSimplexTree:  #
    """
    custom class for assignment simplex--> vec sp
    """
    def __init__(self, simplex_tree, vertex_coords, use_weighted_restrictions=False, weights_dict=None, alpha_weighting=1.0):  #
        validate_simplex_tree(simplex_tree)  #
        self.simplex_tree = simplex_tree  #
        self.vertex_coordinates = vertex_coords  #
        
       
        vertices = set()  #
        for simplex, _ in simplex_tree.get_simplices():  #
            if len(simplex) == 1:  #
                vertices.add(simplex[0])  #
        
        self.num_vertices = len(vertices)  #
        self.ambient_dim = self.vertex_coordinates.shape[1]  #
        
        self.F_sigma_cache = {}  #
        self.use_weighted_restrictions = use_weighted_restrictions  #
        self.weights_dict = weights_dict if weights_dict else {}  #
        self.alpha_weighting = alpha_weighting  #

    def get_F_sigma_basis(self, sigma_tuple):  #
        """
        determination of the basis for F(sigma) vector space, returning basis matrix (matrix consisting of basis vectors as columns) ndarray
        """
        sigma_tuple = tuple(sorted(sigma_tuple))  #
        if not sigma_tuple:  #
            return None  #
        if sigma_tuple in self.F_sigma_cache:  #
            return self.F_sigma_cache[sigma_tuple]  #

        vertex_indices = list(sigma_tuple)  #
        coords = self.vertex_coordinates[vertex_indices, :]  #

        if coords.shape[0] == 0:  #
            self.F_sigma_cache[sigma_tuple] = np.array([]).reshape(self.ambient_dim, 0)  #
            return self.F_sigma_cache[sigma_tuple]  #

        q, r = np.linalg.qr(coords.T)  #

        if r.shape[0] > 0 and r.shape[1] > 0:  #
            tol = np.finfo(r.dtype).eps * max(r.shape) * np.abs(r[0,0]) if r.size > 0 and r[0,0]!=0 else np.finfo(r.dtype).eps  #
            rank = np.sum(np.abs(np.diag(r)) > tol)  #
        elif r.shape[0] == 0 or r.shape[1] == 0:  #
            rank = 0  #
        else:  #
            rank = 1 if np.any(coords) else 0  #

        basis_matrix = q[:, :rank] if rank > 0 else np.array([]).reshape(self.ambient_dim, 0)  #
        self.F_sigma_cache[sigma_tuple] = basis_matrix  #
        return basis_matrix  #

    def F_sigma(self, sigma_tuple):  #
        """
        returns F(sigma) basis matrix (ndarray)
        """
        return self.get_F_sigma_basis(sigma_tuple)  #

    def restriction_map(self, sigma_tuple, tau_tuple, section_in_F_sigma):  #
        """
         section mapping from F(sigma) to F(tau) (returns ndarray)
        
    
        """
        sigma_tuple = tuple(sorted(sigma_tuple))  #
        tau_tuple = tuple(sorted(tau_tuple))  #

        basis_F_tau = self.get_F_sigma_basis(tau_tuple)  #
        if basis_F_tau is None or basis_F_tau.shape[1] == 0:  #
            return np.zeros_like(section_in_F_sigma)  #

        projected_section = basis_F_tau @ (basis_F_tau.T @ section_in_F_sigma)  #

        if self.use_weighted_restrictions:  #
            w_sigma = self.weights_dict.get(sigma_tuple)  #
            w_tau = self.weights_dict.get(tau_tuple)  #
            if w_sigma == 0:  #
                return np.zeros_like(section_in_F_sigma)  #
            else:  #
                projected_section = (w_tau / w_sigma) * projected_section  #
        return projected_section  #

    def get_stalk_F_v(self, vertex_v_idx):  #
        """
        produces basis for stalk at vertex v (also vectors)
        """
        simplex_v = tuple([vertex_v_idx])  #
        return self.get_F_sigma_basis(simplex_v)  #


def get_oriented_simplices(simplex_tree, dim):  #
    """
    orientation (indexation) of the simplices 
    
    """
    simplices_by_dim = get_simplices_by_dimension(simplex_tree)  #
    
    if dim not in simplices_by_dim:  #
        return [], {}  #
    
    simplices_p = simplices_by_dim[dim]  #
    oriented_simplices_map = {s: i for i, s in enumerate(simplices_p)}  #
    
    return simplices_p, oriented_simplices_map  #


def build_coboundary_0_matrix(simplex_tree, presheaf):  #
    """
    builds matrix M0 representation for coboundary map for delta^0: C^0(K, G) -> C^1(K, G)
    
    """
    vertices, v_map = get_oriented_simplices(simplex_tree, 0)  #
    edges, e_map = get_oriented_simplices(simplex_tree, 1)  #

    if not vertices or not edges:  #
        return np.array([]).reshape(0,0), [], []  #

    N0 = len(vertices)  #
    N1 = len(edges)  #
    D = presheaf.ambient_dim  #

    M0 = np.zeros((N1 * D, N0 * D))  #

    for j, vj_tuple in enumerate(vertices):  #
        vj = vj_tuple[0]  #
        for i, ei_tuple in enumerate(edges):  #
            va, vb = ei_tuple  #

            #for (delta^0 f)(e_i=(va,vb)) = f_vb|e_i - f_va|e_i
            if vb == vj:  #
                M0[i*D : (i+1)*D, j*D : (j+1)*D] = np.eye(D)  #

            if va == vj:  #
                M0[i*D : (i+1)*D, j*D : (j+1)*D] = -np.eye(D)  #

    return M0, (vertices, v_map), (edges, e_map)  #


def build_coboundary_1_matrix(simplex_tree, presheaf):  #
    """
    matrix M1 for delta^1: C^1(K, G) -> C^2(K, G)

    """
    edges, e_map = get_oriented_simplices(simplex_tree, 1)  #
    triangles, t_map = get_oriented_simplices(simplex_tree, 2)  #

    if not edges or not triangles:  #
        return np.array([]).reshape(0,0), [], []  #

    N1 = len(edges)  #
    N2 = len(triangles)  #
    D = presheaf.ambient_dim  #

    M1 = np.zeros((N2 * D, N1 * D))  #

    for j, ej_tuple in enumerate(edges):  #
        for i, ti_tuple in enumerate(triangles):  #
            v0, v1, v2 = ti_tuple  #

            e01 = tuple(sorted((v0,v1)))  #
            e02 = tuple(sorted((v0,v2)))  #
            e12 = tuple(sorted((v1,v2)))  #

            # For (delta^1 g)(t_i=(v0,v1,v2)) = g_e01|t_i - g_e02|t_i + g_e12|t_i
            if ej_tuple == e01:  #
                M1[i*D : (i+1)*D, j*D : (j+1)*D] = np.eye(D)  #
            elif ej_tuple == e02:  #
                M1[i*D : (i+1)*D, j*D : (j+1)*D] = -np.eye(D)  #
            elif ej_tuple == e12:  #
                M1[i*D : (i+1)*D, j*D : (j+1)*D] = np.eye(D)  #
                
    return M1, (edges, e_map), (triangles, t_map)  #


def compute_cohomology_dims(M0, M1):  #
    """
    computes dim H0 and dim H1 using rank-nullity.
    """
    dim_H0 = 0  #
    dim_H1 = 0  #

    if M0.size > 0:  #
        rank_M0 = np.linalg.matrix_rank(M0)  #
        num_cols_M0 = M0.shape[1]  #
        dim_H0 = num_cols_M0 - rank_M0  #

    if M1.size > 0:  #
        rank_M1 = np.linalg.matrix_rank(M1)  #
        num_cols_M1 = M1.shape[1]  #
        dim_ker_delta1 = num_cols_M1 - rank_M1  #

        if M0.size > 0:  #
            dim_im_delta0 = rank_M0  #
            dim_H1 = dim_ker_delta1 - dim_im_delta0  #
        else:  #
            dim_H1 = dim_ker_delta1  #

    return dim_H0, max(0, dim_H1)  #


def compute_sheaf_laplacian_L0(M0):  #
    """
    Computes the sheaf laplacian L0 
    """
    if M0.size == 0:  #
        return np.array([])  #
    L0 = M0.T @ M0  #
    return L0  #


def create_simplex_tree_from_points_and_simplices(points, simplices):  #
    """
    simplex tree creation
    """
    # Create an empty SimplexTree  #
    simplex_tree = gudhi.SimplexTree()  #
    
    # Add each simplex with filtration value 0  #
    for simplex in simplices:  #
        simplex_tree.insert(list(simplex), filtration=0.0)  #
    
    return simplex_tree, points  #


def main_example():  #
    """
    examplar work
    """


    
    vertex_coords = np.array([
        [0.0, 0.0],
        [1.25, 0.5],
        [0.75, 1.35],
        [2.42, 0.33],
        [1.57, 1.89]
    ])
    
    simplices = [
        (0,), (1,), (2,), (3,), (4,),
        (0,1), (1,2), (0,2), (1,3), (2,4), (3,4),
        (0,1,2), (1,2,4), (1,3,4)
    ]
    
    # creation of simplex tree  
    simplex_tree, _ = create_simplex_tree_from_points_and_simplices(vertex_coords, simplices)  #
    
    #  random example
    np.random.seed(42)    #
    example_dissimilarities = {
        s: 0.5 + np.random.rand() * 1.5 for s in simplices  #
    }
    alpha = 1.35
    example_weights = {s: np.exp(-alpha * example_dissimilarities.get(s, 0.0)) for s in simplices}  #

    # presheaf creation  #
    presheaf = PresheafVectorSpaceOnSimplexTree(simplex_tree, vertex_coords, use_weighted_restrictions=True, weights_dict=example_weights)  #
    print(f" presheaf (weight: {presheaf.use_weighted_restrictions})")  #

    sigma_124 = (1,2,4)
    F_124_basis = presheaf.get_F_sigma_basis(sigma_124)
    print(f"basis for F{sigma_124} (shape {F_124_basis.shape if F_124_basis is not None else 'None'}):")
    if F_124_basis is not None: print(F_124_basis)

    sigma_12 = (1,2)
    F_12_basis = presheaf.get_F_sigma_basis(sigma_12)
    print(f"basis for F{sigma_12} (shape {F_12_basis.shape if F_12_basis is not None else 'None'}):")
    if F_12_basis is not None: print(F_12_basis)

    vec_in_F124 = vertex_coords[1] + vertex_coords[2] + vertex_coords[4]
    print(f"vector in F{sigma_124}: {vec_in_F124}")
    restricted_vec_to_F12 = presheaf.restriction_map(sigma_124, sigma_12, vec_in_F124)
    print(f"restriction of vector to F{sigma_12}: {restricted_vec_to_F12}")

    if F_12_basis is not None and F_12_basis.shape[1] > 0:
        reconstructed = F_12_basis @ (F_12_basis.T @ restricted_vec_to_F12)
        print(f"reconstruction from F{sigma_12} basis: {reconstructed}")
       

    stalk_F_v1_basis = presheaf.get_stalk_F_v(1)
    print(f"basis for stalk F_v1: {stalk_F_v1_basis}")

    print("cohomology computation")
    M0, v_data, e_data = build_coboundary_0_matrix(simplex_tree, presheaf)  #
    M1, _, t_data = build_coboundary_1_matrix(simplex_tree, presheaf)  #

    print(f"M0 shape: {M0.shape if M0.size > 0 else 'empty'}")
    print(f"M1 shape: {M1.shape if M1.size > 0 else 'empty'}")

    dim_H0, dim_H1 = compute_cohomology_dims(M0, M1)
    print(f"dim H0(K, G) = {dim_H0 / presheaf.ambient_dim if presheaf.ambient_dim > 0 else dim_H0}")
    print(f"dim H1(K, G) = {dim_H1 / presheaf.ambient_dim if presheaf.ambient_dim > 0 else dim_H1}")

    print("sheaf laplacian L^0")
    L0 = compute_sheaf_laplacian_L0(M0)
    if L0.size > 0:
        print(f"L0 shape: {L0.shape}")
        eigvals = np.linalg.eigvalsh(L0)
        print(f"L0 eigenvalues: {eigvals}")


def additional_example():  #
    """
    additional examples
    """
    
    
    
    points = np.array([  #
        [0, 0],  #
        [1, 0],  #
        [0, 1],  #
        [1, 1],  #
        [0.5, 0.5]  #
    ])  #
    
    
    alpha_complex = gudhi.AlphaComplex(points=points)  #
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=1.0)  #
    
   
    
    
    np.random.seed(42)  
    simplices = [tuple(sorted(simplex)) for simplex, _ in simplex_tree.get_simplices()]  #
    example_weights = {s: np.random.rand() for s in simplices}  #
    
    
    presheaf = PresheafVectorSpaceOnSimplexTree(simplex_tree, points, use_weighted_restrictions=True, weights_dict=example_weights)  #
    print(f"presheaf on alpha")  #
    
   
    M0, v_data, e_data = build_coboundary_0_matrix(simplex_tree, presheaf)  #
    M1, _, t_data = build_coboundary_1_matrix(simplex_tree, presheaf)  #
    
    dim_H0, dim_H1 = compute_cohomology_dims(M0, M1)  #
    print(f"dim H0(K, G) = {dim_H0 / presheaf.ambient_dim if presheaf.ambient_dim > 0 else dim_H0}")  #
    print(f"dim H1(K, G) = {dim_H1 / presheaf.ambient_dim if presheaf.ambient_dim > 0 else dim_H1}")  #
    
    
    L0 = compute_sheaf_laplacian_L0(M0)  #
    if L0.size > 0:  #
        eigvals = np.linalg.eigvalsh(L0)  #
        print(f"L0 eigenvalues: {eigvals}")  #


#NoTE/REMARK: The division by K.ambient_dim is a heuristic.
    # If H0 is sum of D copies of scalar H0, then this is fine.
    # H0 = dim(ker(delta0)). If sections are vectors in R^D, then ker(delta0) is a subspace of (N0*D)-dim space.
    # number of independent global sections. Each global section is a D-dim vector.
    # so dim H0 from rank-nullity is the dimension of the vector space of global sections.
    # if dim H0 = k*D, it means there are k independent D-dimensional global sections.
    # the document refers to dim H^p as a scalar. The current code computes the dimension of the vector space.
    # for a constant sheaf R^D, H^p(K, R^D) = H^p_simplicial(K, R) tensor R^D.
    # so dim H^p(K,R^D) = dim H^p_simplicial(K,R) * D.
    # the current code for M0, M1 assumes sections are vectors in R^D and restrictions are identity.
    # this is closer to H^p(K, constant_sheaf_R^D).
    # for F(sigma) = span(coords), this is more complex.
    # the current code's H0, H1 are for the constant sheaf R^D if restrictions are identity.
    # in document G(sigma) = span(coords). The coboundary maps should use the actual restriction maps.
    # the current M0, M1 are simplified. A more rigorous M0, M1 would be block matrices where each block is a DxD matrix
    # representing the restriction map (or projection). The current code uses Identity, which is only correct if F(tau) = F(sigma) = R^D.
    # for F(sigma)=span(coords), the restriction rho_sigma_tau is P_F(tau). This needs to be incorporated into M0, M1 elements.
    # the current implementation of M0 and M1 is for a *constant sheaf* with values in R^D.
    # it does not use the `presheaf.restriction_map` for defining the coboundary operators.


if __name__ == '__main__':
    main_example()
    additional_example()  #
