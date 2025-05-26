import csv
import numpy as np
import pandas as pd
import TDA
from scipy.spatial.distance import cdist, directed_hausdorff
from ripser import ripser
from persim import plot_diagrams


def create_covariance_matrix(n):
    with open("randommatrix.py", "w") as csvfile:
        data = np.random.rand(n,n)
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)


def remove_impacted_regions(data_csv, indicator_csv, output_csv):
    # Load data matrix
    data_df = pd.read_csv(data_csv, index_col=0)

    # Load indicator matrix
    indicator_df = pd.read_csv(indicator_csv, index_col=0)

    # Extract columns where indicator is 1
    included_regions = indicator_df.columns[indicator_df.iloc[0] == 1]

    # Subset data matrix
    filtered_df = data_df.loc[included_regions, included_regions]

    # Save result
    i = filtered_df.to_csv(output_csv)

    return

def remove_unimpacted_regions(data_csv, indicator_csv, output_csv):
    # Load data matrix
    data_df = pd.read_csv(data_csv, index_col=0)

    # Load indicator matrix
    indicator_df = pd.read_csv(indicator_csv, index_col=0)

    # Extract columns where indicator is 1
    included_regions = indicator_df.columns[indicator_df.iloc[0] == 0]

    # Subset data matrix
    filtered_df = data_df.loc[included_regions, included_regions]

    # Save result
    i = filtered_df.to_csv(output_csv)

    return

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
        return 0.0
    if point_cloud1.size == 0 or point_cloud2.size == 0:
        return float('inf')

    if underlying_metric == 'euclidean' and (metric_params is None or metric_params == {}):
        h_a_b, _, _ = directed_hausdorff(point_cloud1, point_cloud2)
        h_b_a, _, _ = directed_hausdorff(point_cloud2, point_cloud1)
        return max(h_a_b, h_b_a)
    else:
        if callable(underlying_metric):
            pairwise_dist_matrix = cdist(point_cloud1, point_cloud2, metric=underlying_metric)
        elif isinstance(underlying_metric, str):
            params = metric_params if metric_params is not None else {}
            pairwise_dist_matrix = cdist(point_cloud1, point_cloud2, metric=underlying_metric, **params)
        else:
            return float('inf')

        # h(A, B) = sup_{a in A} min_{b in B} d(a, b)
        min_dists_c1_to_c2 = np.min(pairwise_dist_matrix, axis=1)
        h_a_b = np.max(min_dists_c1_to_c2)
        min_dists_c2_to_c1 = np.min(pairwise_dist_matrix, axis=0)
        h_b_a = np.max(min_dists_c2_to_c1)
        
        return max(h_a_b, h_b_a)


if __name__ == "__main__":
    idd="1"

    prepath = "/Users/douglascook/Desktop/Pre-Post-csv/Joe Data/Pre_"+idd+"_covariance.csv"
    salvaged = "/Users/douglascook/Downloads/salvaged.csv"

    output_path_pre = "/Users/douglascook/Downloads/filtered_Pre_"+idd+"_covariance.csv"

    remove_impacted_regions(prepath, salvaged, output_path_pre)

    postpath = "/Users/douglascook/Desktop/Pre-Post-csv/Joe Data/Post_"+idd+"_covariance.csv"

    output_path_post = "/Users/douglascook/Desktop/Pre-Post-csv/Joe Data/filtered_Post_"+idd+"_covariance.csv"

    remove_impacted_regions(postpath, salvaged, output_path_post)

    unimpacted_pre_cor = TDA.cov_to_cor(output_path_pre)
    unimpacted_post_cor = TDA.cov_to_cor(output_path_post)

    unimpacted_pre_dist = TDA.cor_to_dist(unimpacted_pre_cor)
    unimpacted_post_dist = TDA.cor_to_dist(unimpacted_post_cor)

    unimpacted_pointCloudPre = TDA.classical_mds(unimpacted_pre_dist)
    unimpacted_pointCloudPost = TDA.classical_mds(unimpacted_post_dist)

    print("Hausdorff Distance for unimpacted regions: ", compute_hausdorff_distance(unimpacted_pointCloudPre, unimpacted_pointCloudPost))

    output_path_pre = "/Users/douglascook/Downloads/filtered_Pre_"+idd+"_covariance.csv"

    remove_unimpacted_regions(prepath, salvaged, output_path_pre)

    postpath = "/Users/douglascook/Desktop/Pre-Post-csv/Joe Data/Post_"+idd+"_covariance.csv"

    output_path_post = "/Users/douglascook/Desktop/Pre-Post-csv/Joe Data/filtered_Post_"+idd+"_covariance.csv"

    remove_unimpacted_regions(postpath, salvaged, output_path_post)

    impacted_pre_cor = TDA.cov_to_cor(output_path_pre)
    impacted_post_cor = TDA.cov_to_cor(output_path_post)

    impacted_pre_dist = TDA.cor_to_dist(impacted_pre_cor)
    impacted_post_dist = TDA.cor_to_dist(impacted_post_cor)

    impacted_pointCloudPre = TDA.classical_mds(impacted_pre_dist)
    impacted_pointCloudPost = TDA.classical_mds(impacted_post_dist)

    print("Hausdorff Distance for impacted regions: ", compute_hausdorff_distance(impacted_pointCloudPre, impacted_pointCloudPost))

    # 4 obtained point clouds; unimpacted_pointCloudPre, unimpacted_pointCloudPost, impacted_pointCloudPre, impacted_pointCloudPost

    diagram = ripser(impacted_pointCloudPre)['dgms']
    plot_diagrams(diagram, show=True)

    diagram = ripser(impacted_pointCloudPost)['dgms']
    plot_diagrams(diagram, show=True)    

    diagram = ripser(unimpacted_pointCloudPre)['dgms']
    plot_diagrams(diagram, show=True)

    diagram = ripser(unimpacted_pointCloudPost)['dgms']
    plot_diagrams(diagram, show=True)    


    