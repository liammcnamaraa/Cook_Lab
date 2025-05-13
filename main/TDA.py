from os import listdir
from os.path import isfile, join
import gudhi.delaunay_complex
import numpy as np
import pandas as pd
import csv
from ripser import ripser
from persim import plot_diagrams
from persim import bottleneck
from persim import sliced_wasserstein
import matplotlib.pyplot as plt
import gudhi
from sklearn.manifold import MDS
from scipy.spatial.distance import minkowski
from itertools import combinations
from numpy.linalg import eigh, norm
from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, BettiCurve, PairwiseDistance

# global declarations
p = np.inf


def betti_curve(dgm, window_number=3):
    BC = BettiCurve()

    X_betti_curves = BC.fit_transform(dgm)

    BC.plot(X_betti_curves, sample=window_number)


def cov_to_cor(path: str):
    df = pd.read_csv(path, index_col=0)
    covariance_matrix = df.to_numpy()
    num_parameters=len(covariance_matrix)   
    correlation_matrix = np.zeros((num_parameters, num_parameters), dtype=float)

    i = 0
    while (i < num_parameters):
        j = 0
        while (j < num_parameters):
            std_i = np.sqrt(covariance_matrix[i][i])
            std_j = np.sqrt(covariance_matrix[j][j])
            if std_i == 0 or std_j == 0:
                correlation_matrix[i][j] = 0  
            else:
                correlation_matrix[i][j] = covariance_matrix[i][j] / (std_i * std_j)

            j+=1            
        i+=1

    return correlation_matrix


def cor_to_dist(correlation_matrix):
    num_parameters=len(correlation_matrix)   
    ones = np.ones((num_parameters, num_parameters))
    dist_matrix = ones - correlation_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    
    for i in range(num_parameters):
        for j in range(num_parameters):
            if i != j:
                dist_matrix[i, j] = np.sqrt(abs(2 * (1 - dist_matrix[i, j])))
            else:
                continue
    return dist_matrix

def p_plot(simplex_tree):
    pairs = simplex_tree.persistence()
    if pairs:
        gudhi.plot_persistence_diagram(pairs, legend=True)
        plt.title("Alpha Complex Persistence Diagram")
        plt.show()

        gudhi.plot_persistence_barcode(pairs, legend=True)
        plt.title("Alpha Complex Persistence Barcode")
        plt.show()


def frobenius_norm(dist_matrix):
    # frobenius:= sqrt(Tr(A^T \times A)) 
    matrix = dist_matrix.transpose() * dist_matrix
    size = matrix.shape[0]
    
    trace = 0
    i = 0
    while i < size:
        trace += matrix[i][i]

    return np.sqrt(trace)

def get_diagrams(dir: str):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    files.sort()
    
    dgms = {}
    for file in files:
        correlation_m = cov_to_cor(join(dir,file))
        dist_m = cor_to_dist(correlation_m)
        diagrams = ripser(dist_m, distance_matrix=True)['dgms']
        dgms[file] = diagrams
    
    print("diagrams generated...")
    return dgms

def compare_h1_dgms(dgm1_, dgm2_):
    dgm1 = dgm1_[1]
    dgm2 = dgm2_[1]

    b_dist = bottleneck(dgm1,dgm2)
    sw_dist = sliced_wasserstein(dgm1,dgm2,M=50)

    return [b_dist, sw_dist]

def construct_cech_complex(data):
    return

def construct_alpha_complex(data):
    return


def kashtanov_mds(dist_matrix, n_components=3):
  n = dist_matrix.shape[0]
  H = np.eye(n) - np.ones((n,n)) / n

  dist_squared = dist_matrix ** 2
  B = -0.5 * H @ dist_squared @ H

  eigvals, eigvecs = np.linalg.eigh(B)
  idx = np.argsort(eigvals)[::-1]
  eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

  L = np.diag(np.sqrt(np.maximum(eigvals[:n_components], 0)))
  V = eigvecs[:, :n_components]
  X = V @ L

  return X

def some_code(point_cloud):
    # ---------------- 1. Build the Alpha complex ----------------
    # In GUDHI, `create_simplex_tree` for AlphaComplex expects `max_alpha_square`,
    # i.e. the *square* of the alpha‑radius you want to truncate at.
    max_alpha          = 1.0          # ADJUSTABLE  (units = distance)
    max_alpha_square   = max_alpha**2 # what the API needs

    alpha_complex = gudhi.AlphaComplex(points=point_cloud)
    simplex_tree  = alpha_complex.create_simplex_tree(max_alpha_square=max_alpha_square)

    print(f"\nAlpha complex created with "
          f"{simplex_tree.num_simplices()} simplices and "
          f"{simplex_tree.num_vertices()} vertices "
          f"(α ≤ {max_alpha}).")
    
    # ---------------- Persistent homology --------------------
    homology_dimensions = [0, 1, 2]   # ADJUSTABLE
    min_persistence     = 0.15        # ADJUSTABLE
    coeff_field         = 3           # ADJUSTABLE (prime)

    simplex_tree.persistence(homology_coeff_field=coeff_field,
                             min_persistence=min_persistence)

    # pull out a few dimensions for printing
    for dim in homology_dimensions:
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        if intervals.size:
            print(f"\nPersistence intervals for H{dim}:")
            for birth, death in intervals:
                death_str = f"{death:.2f}" if np.isfinite(death) else "∞"
                print(f"  birth {birth:.2f}, death {death_str}")
        else:
            print(f"\nNo intervals in H{dim} (or not computed).")

    # plot persistence
    p_plot(simplex_tree=simplex_tree)


def bn_MDS(dist_matrix, k=2):
    N = dist_matrix.shape[0]

    mds = MDS ( n_components =k, dissimilarity ='precomputed ', random_state =42, n_init =4, max_iter =300)
    point_cloud = mds.fit_transform(dist_matrix)

    return point_cloud


def chebyshev_distance(point1, point2):
    p1 = np.asarray(point1)
    p2 = np.asarray(point2)
    dist = np.max(np.abs(p1 - p2))

    return dist


def general_lp_distance(point1, point2, p_norm_value):
    """ here, i tried firstly implement L^p metric for construction of Cech complexes but it was very slow. if you know how to solve it, help me out
   . it can be proven that if p-->infinity, then L^p--> L^infinity """
    if p_norm_value == np.inf:
        return chebyshev_distance(point1, point2)
    elif p_norm_value < 1:
        raise ValueError("p_norm_value must be >= 1 for Lp distance.")
    # scipy.spatial.distance.minkowski computes the L^p nor
    return minkowski(point1, point2, p_norm_value)


def get_min_enclosing_chebyshev_ball_radius(points_for_simplex):
    """this computes the radius of the minimum enclosing L^infinity ball
    for a given set of points (vertices of a simplex).

    The Cech filtration value for a simplex is the radius of its minimum enclosing ball.

    """
    if not isinstance(points_for_simplex, np.ndarray):
        points_array = np.array(points_for_simplex)
    else:
        points_array = points_for_simplex

    if points_array.size == 0:
        return 0.0
    if points_array.ndim == 1:
        # this case handles a single point-vertex (0-simplex), so the radius is 0
        points_array = points_array.reshape(1, -1)

    # for a set of points, the minimum enclosing Chebyshev ball
    # has its center c_j = (min_coord_j + max_coord_j) / 2 for each dimension j.
    # the radius R is max_j ((max_coord_j - min_coord_j) / 2).
    min_coords = np.min(points_array, axis=0)
    max_coords = np.max(points_array, axis=0)

    # the diameter along each dimension is (max_coords - min_coords).
    # the radius contribution from each dimension is half of this diameter.
    radius = general_lp_distance(max_coords, min_coords, p) / 2.0
    return radius


def construct_chebyshev_cech_complex(point_cloud, max_simplex_dim, max_filtration_val=None):
    """ so, as Gudhi has already built-in for construction Cech complexes but it uses L2 (standard Euclidean metric)
    The only problem here is that L^infinity returns the maximum of the absolute value of entry-wise difference
    for tuple/point/vertex coordinates. So, it is better to use the L^p metric but my tries were not successful
    """
    num_points = point_cloud.shape[0]
    # initialize an empty simplex tree using gudhi built-in
    simplex_tree = gudhi.SimplexTree()

    # Add 0-simplices (vertices)
    # Each point in the cloud is a 0-simplex. These are typically considered to appear at filtration value 0.
    for i in range(num_points):
        # A 0-simplex [i] (vertex i) has a minimum enclosing ball of radius 0 centered at itself.
        simplex_tree.insert([i], filtration=0.0)
    for dim_k_minus_1 in range(1, max_simplex_dim + 1):
        num_vertices_in_simplex = dim_k_minus_1 + 1
        for simplex_indices_tuple in combinations(range(num_points), num_vertices_in_simplex):
            simplex_indices = list(simplex_indices_tuple) # Convert tuple to list for indexing
            current_simplex_points = point_cloud[simplex_indices, :]
            filtration_value = get_min_enclosing_chebyshev_ball_radius(current_simplex_points)

            # add the simplex to the tree if its filtration value is within the allowed maximum (if specified).
            if max_filtration_val is None or filtration_value <= max_filtration_val:
                # The insert method of Gudhi's SimplexTree handles the requirement that faces
                # of a simplex must exist with smaller or equal filtration values.
                # Our construction order (increasing dimension) and the property of enclosing ball radii
                # (radius for a face <= radius for the simplex) ensure this is valid.
                simplex_tree.insert(simplex_indices, filtration=filtration_value)

    return simplex_tree


def compute_persistence_homology(simplex_tree, min_persistence_val=0.0):
    """
    """
    # Ensure the simplex tree has its filtration values computed and simplices are ordered.
    # This is usually done implicitly if constructing manually with filtration values,
    # but calling it explicitly can be good practice if the tree was modified.
    # simplex_tree.ensure_coherence() # Optional, might be useful in complex scenarios.

    # ADJUSTABLE: homology_coeff_field determines the field for homology computation.
    # Common choices: 2 for Z/2Z (most common, simplest), or a prime p for Z/pZ.
    # Gudhi uses integer codes for fields (e.g., 11 for Z/11Z).
    coeff_field = 2 # Using Z/2Z field coefficients. LATER WE CAN ADJUST IT

    # Compute the persistence pairs.
    # The `persistence()` method calculates the birth and death times of topological features.
    persistence_pairs = simplex_tree.persistence(homology_coeff_field=coeff_field, min_persistence=min_persistence_val)

    # The output `persistence_pairs` is a list of tuples:
    # (dimension_of_feature, (birth_time, death_time))
    # For 0-dimensional homology (connected components), one component (the first one to appear)
    # might live infinitely if the complex remains connected. Gudhi represents infinite death time
    # as float('inf') or a very large double, depending on the version/context.

    return persistence_pairs

def _inv_sqrt(mat, eps=1e-10):
    """
    Return A^{-1/2} for a symmetric positive-definite matrix A
    using an eigen-decomposition.  A small `eps` guards against
    tiny or negative eigenvalues caused by round-off.
    """
    eigval, eigvec = eigh(mat)
    eigval = np.clip(eigval, eps, None)        # regularise
    inv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
    return inv_sqrt, eigval.min()              # also return min λ for diagnostics

def correlation_to_distance(C, alpha=1.0, eps=1e-8):
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square 2-D array")
    N = C.shape[0]

    # Pre-compute all P_i and their inverse square roots
    Pinv_sqrts = []
    for i in range(N):
        v = C[i]                               # row vector v_i
        P_i = np.eye(N) + alpha * np.outer(v, v)
        P_inv_sqrt, _ = _inv_sqrt(P_i, eps)
        Pinv_sqrts.append(P_inv_sqrt)

    # Allocate the distance matrix
    D = np.zeros((N, N), dtype=float)

    # Double loop over (i, j) with symmetry
    for i in range(N):
        P_inv_sqrt_i = Pinv_sqrts[i]
        for j in range(i + 1, N):
            # Affine-invariant distance
            M = P_inv_sqrt_i @ (np.eye(N) + alpha *
                                np.outer(C[j], C[j])) @ P_inv_sqrt_i
            # log(M) via eigen-decomposition
            eigval, eigvec = eigh(M)
            eigval = np.clip(eigval, eps, None)
            log_M = eigvec @ np.diag(np.log(eigval)) @ eigvec.T
            D[i, j] = D[j, i] = norm(log_M, ord='fro')

    return D





if __name__ == '__main__':
    path = "/Users/douglascook/Cook_Lab/data/joe_data"
    #d = get_diagrams(path)
    #d1=d["Pre_1_covariance.csv"]
    #d2=d["Post_1_covariance.csv"]

    #print(compare_h1_dgms(d1, d2))
    cor = cov_to_cor("data/joe_data/Post_0_covariance.csv")
    print("Correlation matrix generated...")
    dist = cor_to_dist(cor)
    print("Distance matrix generated")
    print(dist)
    pc = kashtanov_mds(dist)
    print("Point cloud generated")
    some_code(pc)
    
    