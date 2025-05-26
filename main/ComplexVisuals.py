import gudhi as gd
import matplotlib.pyplot as plt
import gudhi.representations as gr
import numpy as np
import TDA


if __name__ == "__main__":
    print("Start")

    # Obtain Point Cloud 
    filePath = "/Users/douglascook/Desktop/Pre-Post-csv/Joe Data/Post_0_covariance.csv"
    correlation = TDA.cov_to_cor(filePath)
    distance = TDA.cor_to_dist(correlation)
    pointCloud = TDA.classical_mds(distance)

    print("Obtained point cloud")

    # Generate Rips Complex
   # ripsComplex = gd.RipsComplex(points=pointCloud, max_edge_length=2.0)
    #st_rips = ripsComplex.create_simplex_tree(max_dimension=3)

    #print("rips generated")

    # Generate Alpha Complex
    alphaComplex = gd.AlphaComplex(points=pointCloud)
    st_alpha = alphaComplex.create_simplex_tree()

    print("alpha generated")

    # Generate Cech Complex
    #cechComplex = TDA.construct_chebyshev_cech_complex(point_cloud=pointCloud, max_simplex_dim=2)
    #st_cech = cechComplex.create_simplex_tree(max_dimenstion=3)

    #print("cech generated")

    # Visualize Simplical Complexes
    #rips_edges = st_rips.get_skeleton(3)


    # Compute Persistence
    #st_rips.compute_persistence()
    #st_alpha.compute_persistence()
    #st_cech.compute_persistence()

    print("persistence computed")

    # Display Persistence Diagrams
    #gd.plot_persistence_diagram(st_rips.persistence())
    #plt.title("Rips Persistence Diagram")
    #plt.show()

    #gd.plot_persistence_diagram(st_alpha.persistence())
    #plt.title("Alpha Persistence Diagram")
    #plt.show()

    #gd.plot_persistence_diagram(st_cech.persistence())
    #plt.title("Cech Persistence Diagram")
    #plt.show()

    print("Done")