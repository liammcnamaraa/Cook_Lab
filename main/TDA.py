from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import csv
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt


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
    return ones - correlation_matrix


def get_diagrams(dir: str):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    files.sort()
    
    dgms = {}
    for file in files:
        correlation_m = cov_to_cor(join(dir,file))
        dist_m = cor_to_dist(correlation_m)
        diagrams = ripser(dist_m, distance_matrix=True)['dgms']
        dgms[file] = diagrams
    
    for i in range(0, 4):
        num = str(i)
        plot_diagrams([dgms['Pre_'+num+'_covariance.csv'][1], dgms['Post_'+num+'_covariance.csv'][1]], labels=["Pre", "Post"], show=True)


if __name__ == '__main__':
    path = "/Users/douglascook/Cook_Lab/data/joe_data"
    get_diagrams(path)