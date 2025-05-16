"""
Code for the computation of the Gromov-Hausdorff and Hausdorff metrics (distances)
"""

import numpy as np
from scipy.spatial.distance import cdist, directed_hausdorff
from itertools import combinations
from scipy.optimize import linear_sum_assignment


def represent_simplicial_complex(simplices, vertex_coordinates):
    """
    Represents a simplicial complex with its vertex coordinates in dictionary. It will help later to obtain the affine hulls of the point
    cloud data. So the args are arrays and this func returns the dictionary (typical presentation of a graph as list of vertices and edges). the output is the dictionary
    """
    return {
        "simplices": simplices,
        "vertex_coordinates": vertex_coordinates
    }


def get_affine_hull_points(simplex_vertices, num_samples_per_simplex=100):
    """
    Takes array of vertex coordinates for k-simplex and returns sampled points from simplex's convex hull.
    """
    num_vertices_in_simplex, dim = simplex_vertices.shape

    if num_vertices_in_simplex == 0:
        return np.array([]).reshape(0, dim)
    if num_vertices_in_simplex == 1:
        return simplex_vertices.copy()

    bary_coords = np.random.dirichlet(np.ones(num_vertices_in_simplex), size=num_samples_per_simplex)
    sampled_points = bary_coords @ simplex_vertices
    return sampled_points


def geometric_realization_point_cloud(simplicial_complex, num_samples_per_simplex=100, min_simplex_dim=0):
    """
    Computes the geometric realization of a simplex complex.
    """
    simplices = simplicial_complex["simplices"]
    vertex_coords = simplicial_complex["vertex_coordinates"]
    
    all_sampled_points = []

    if not simplices:
        if vertex_coords.shape[0] > 0 and min_simplex_dim == 0:
             return vertex_coords.copy()
        return np.array([]).reshape(0, vertex_coords.shape[1])

    for simplex_indices in simplices:
        if not simplex_indices:
            continue
        
        simplex_dim = len(simplex_indices) - 1
        if simplex_dim < min_simplex_dim:
            continue
            
        current_simplex_vertices = vertex_coords[simplex_indices, :]
        
        if current_simplex_vertices.shape[0] == 0:
            continue
        n_samples = num_samples_per_simplex
        if simplex_dim == 0 and n_samples == 0:
            n_samples = 1 
        elif simplex_dim == 0 and n_samples > 0:
            n_samples = 1
        elif n_samples == 0:
            continue

        points_on_simplex = get_affine_hull_points(current_simplex_vertices, num_samples_per_simplex=n_samples)
        if points_on_simplex.size > 0:
            all_sampled_points.append(points_on_simplex)
    
    if not all_sampled_points:
        if min_simplex_dim == 0 and any(len(s) == 1 for s in simplices):
            zero_simplices_indices = [s[0] for s in simplices if len(s) == 1]
            if zero_simplices_indices:
                return vertex_coords[list(set(zero_simplices_indices)), :]
        return np.array([]).reshape(0, vertex_coords.shape[1])
        
    return np.vstack(all_sampled_points)


def compute_hausdorff_distance(
    point_cloud1, 
    point_cloud2, 
    underlying_metric='euclidean', 
    metric_params=None
):
    """
    Computes the Hausdorff distance between two geometric realizations.
    H(A, B) = max(h(A, B), h(B, A)), where h(A, B) = sup_{a in A} inf_{b in B} d(a, b).
    """
    if point_cloud1.size == 0 and point_cloud2.size == 0:
        return 0.0, point_cloud1, point_cloud2
    if point_cloud1.size == 0 or point_cloud2.size == 0:
        return float('inf'), point_cloud1, point_cloud2

    if underlying_metric == 'euclidean' and (metric_params is None or metric_params == {}):
        h_a_b, _, _ = directed_hausdorff(point_cloud1, point_cloud2)
        h_b_a, _, _ = directed_hausdorff(point_cloud2, point_cloud1)
        return max(h_a_b, h_b_a), point_cloud1, point_cloud2
    else:
        if callable(underlying_metric):
            pairwise_dist_matrix = cdist(point_cloud1, point_cloud2, metric=underlying_metric)
        elif isinstance(underlying_metric, str):
            params = metric_params if metric_params is not None else {}
            pairwise_dist_matrix = cdist(point_cloud1, point_cloud2, metric=underlying_metric, **params)
        else:
            return float('inf'), point_cloud1, point_cloud2

        # h(A, B) = sup_{a in A} min_{b in B} d(a, b)
        min_dists_c1_to_c2 = np.min(pairwise_dist_matrix, axis=1)
        h_a_b = np.max(min_dists_c1_to_c2)
        min_dists_c2_to_c1 = np.min(pairwise_dist_matrix, axis=0)
        h_b_a = np.max(min_dists_c2_to_c1)
        
        return max(h_a_b, h_b_a), point_cloud1, point_cloud2


def compute_gromov_hausdorff_distance(
    point_cloud1, 
    point_cloud2, 
    metric1='euclidean',
    metric1_params=None,
    metric2='euclidean',
    metric2_params=None
):
    """
    Computes an approximation of the Gromov-Hausdorff distance between two geometric realizations.
    """
    if point_cloud1.size == 0 and point_cloud2.size == 0:
        return 0.0
    if point_cloud1.size == 0 or point_cloud2.size == 0:
        return float('inf')

    # Calculate the distance matrices for each point cloud
    params1 = metric1_params if metric1_params is not None else {}
    if callable(metric1):
        C1 = cdist(point_cloud1, point_cloud1, metric=metric1)
    else:
        C1 = cdist(point_cloud1, point_cloud1, metric=metric1, **params1)

    params2 = metric2_params if metric2_params is not None else {}
    if callable(metric2):
        C2 = cdist(point_cloud2, point_cloud2, metric=metric2)
    else:
        C2 = cdist(point_cloud2, point_cloud2, metric=metric2, **params2)

    # ensuring the symmetry and zero diagonal
    C1 = np.maximum(C1, C1.T)
    C2 = np.maximum(C2, C2.T)
    np.fill_diagonal(C1, 0)
    np.fill_diagonal(C2, 0)

    # then we use Memoli approximation algorithm. we fistly handle the size differences here so we need to pad the smaller matrix
    n1, n2 = C1.shape[0], C2.shape[0]
    
    # ff the shapes are different we pad the smaller one
    if n1 < n2:
        pad_size = n2 - n1
        C1_padded = np.pad(C1, ((0, pad_size), (0, pad_size)), mode='constant')
        C2_padded = C2.copy()
    elif n2 < n1:
        pad_size = n1 - n2
        C2_padded = np.pad(C2, ((0, pad_size), (0, pad_size)), mode='constant')
        C1_padded = C1.copy()
    else:
        C1_padded = C1.copy()
        C2_padded = C2.copy()
    
    # cost matrix computation
    n = max(n1, n2)
    cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            diffs = []
            for k in range(n):
                for l in range(n):
                    diffs.append(abs(C1_padded[i, k] - C2_padded[j, l]))
            cost_matrix[i, j] = max(diffs) if diffs else 0
    
    # using the hungarian algorithm 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # computing the GH metric approximation
    gh_approx = cost_matrix[row_ind, col_ind].max() / 2
    
    return gh_approx


def main_example():
    print("Computaiton of H and GH metrics")

    vertex_coords_pre = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],  
        [0.5, 0.5, 1],  
        [2, 0, 0], [2, 1, 0],  
        [1.5, 0.5, 0.8]  
    ])


    simplices_pre = [
        [0], [1], [2], [3], [4], [5], [6], [7],  
        [0, 1], [1, 3], [3, 2], [2, 0],  
        [0, 4], [1, 4], [2, 4], [3, 4],  
        [5, 6], [5, 7], [6, 7],  
        [0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4],  
        [5, 6, 7]  
    ]

    complex_pre = represent_simplicial_complex(simplices_pre, vertex_coords_pre)
    print("\nPre-complex defined.")

    
    vertex_coords_post = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],  
        [0.5, 0.5, 0.9],  
        [2.1, 0.1, 0.1], [2.1, 1.1, 0.1],  
        [1.6, 0.6, 0.75]  
    ])

    simplices_post = [
        [0], [1], [2], [3], [4], [5], [6], [7],  
        [0, 1], [1, 3], [3, 2], [2, 0],  
        [0, 4], [1, 4], [3, 4],  
        [5, 6], [5, 7], [6, 7],  
        [0, 1, 4], [1, 3, 4], [3, 4, 2],  
        [5, 6, 7]  
    ]

    complex_post = represent_simplicial_complex(simplices_post, vertex_coords_post)
    print("Post-complex defined.")

    
    num_samples_simplex = 20  
    min_dim_realization = 0
    print(f"\nGenerating geometric realizations (point clouds) with {num_samples_simplex} samples/simplex, min_dim={min_dim_realization}...")
    
    realization_pre = geometric_realization_point_cloud(complex_pre, 
                                                       num_samples_per_simplex=num_samples_simplex, 
                                                       min_simplex_dim=min_dim_realization)
    realization_post = geometric_realization_point_cloud(complex_post, 
                                                        num_samples_per_simplex=num_samples_simplex,
                                                        min_simplex_dim=min_dim_realization)
    print(f"Pre-complex realization: {realization_pre.shape[0]} points.")
    print(f"Post-complex realization: {realization_post.shape[0]} points.")

    
    max_points = 50  
    if realization_pre.shape[0] > max_points:
        indices = np.random.choice(realization_pre.shape[0], max_points, replace=False)
        realization_pre = realization_pre[indices]
        print(f"Sampled down to {realization_pre.shape[0]} points for pre-complex.")
    
    if realization_post.shape[0] > max_points:
        indices = np.random.choice(realization_post.shape[0], max_points, replace=False)
        realization_post = realization_post[indices]
        print(f"Sampled down to {realization_post.shape[0]} points for post-complex.")

    
    print("\nComputing Hausdorff Distance...")
    hausdorff_dist_euclidean, _, _ = compute_hausdorff_distance(realization_pre, realization_post, underlying_metric='euclidean')
    print(f"  Hausdorff Distance (euclidean/L^2): {hausdorff_dist_euclidean:.4f}")

    hausdorff_dist_manhattan, _, _ = compute_hausdorff_distance(realization_pre, realization_post, underlying_metric='cityblock')
    print(f"  Hausdorff Distance (L^1): {hausdorff_dist_manhattan:.4f}")

    hausdorff_dist_chebyshev, _, _ = compute_hausdorff_distance(realization_pre, realization_post, underlying_metric='chebyshev')
    print(f"  Hausdorff Distance (chebyshev/L^inf): {hausdorff_dist_chebyshev:.4f}")

    hausdorff_dist_minkowski_p3, _, _ = compute_hausdorff_distance(realization_pre, realization_post, 
                                                                 underlying_metric='minkowski', 
                                                                 metric_params={'p': 3})
    print(f"  Hausdorff Distance (minkowski L^3): {hausdorff_dist_minkowski_p3:.4f}")

    
    print("\nComputing Gromov-Hausdorff Distance...")
    
    
    gh_dist_euclidean = compute_gromov_hausdorff_distance(
        realization_pre, realization_post,
        metric1='euclidean', metric2='euclidean'
    )
    print(f"  Gromov-Hausdorff Distance (euclidean/L^2): {gh_dist_euclidean:.4f}")
    
    
    gh_dist_manhattan = compute_gromov_hausdorff_distance(
        realization_pre, realization_post,
        metric1='cityblock', metric2='cityblock'
    )
    print(f"  Gromov-Hausdorff Distance (L^1): {gh_dist_manhattan:.4f}")
    
    
    gh_dist_chebyshev = compute_gromov_hausdorff_distance(
        realization_pre, realization_post,
        metric1='chebyshev', metric2='chebyshev'
    )
    print(f"  Gromov-Hausdorff Distance (chebyshev/L^inf): {gh_dist_chebyshev:.4f}")
    
    
    gh_dist_minkowski_p3 = compute_gromov_hausdorff_distance(
        realization_pre, realization_post,
        metric1='minkowski', metric1_params={'p': 3},
        metric2='minkowski', metric2_params={'p': 3}
    )
    print(f"  Gromov-Hausdorff Distance (minkowski L^3): {gh_dist_minkowski_p3:.4f}")


if __name__ == '__main__':
    main_example()
