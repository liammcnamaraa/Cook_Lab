import TDA
import newHandGHmetrics as nhg
import gudhi as gd


def dmitrii_kashtanov_konstantinovich():
    path1 = "/Users/douglascook/Desktop/Pre-Post-csv/Joe Data/Pre_0_covariance.csv"
    path2 = "/Users/douglascook/Desktop/Pre-Post-csv/Joe Data/Post_0_covariance.csv"

    cor1 = TDA.cov_to_cor(path1)
    cor2 = TDA.cov_to_cor(path2)

    dist1 = TDA.cor_to_dist(cor1)
    dist2 = TDA.cor_to_dist(cor2)

    pointCloud1 = TDA.classical_mds(dist1)
    pointCloud2 = TDA.classical_mds(dist2)

    print("Obtained point clouds")

    #st_cech1 = TDA.construct_chebyshev_cech_complex(point_cloud=pointCloud1, max_simplex_dim=3)
    #st_cech2 = TDA.construct_chebyshev_cech_complex(point_cloud=pointCloud2, max_simplex_dim=3)

    print("cech done")

    ripsComplex1 = gd.RipsComplex(points=pointCloud1, max_edge_length=2.0)
    st_rips1 = ripsComplex1.create_simplex_tree(max_dimension=3)

    ripsComplex2 = gd.RipsComplex(points=pointCloud2, max_edge_length=2.0)
    st_rips2 = ripsComplex2.create_simplex_tree(max_dimension=3)

    print("rips done")

    alphaComplex1 = gd.AlphaComplex(points=pointCloud1)
    st_alpha1 = alphaComplex1.create_simplex_tree()

    alphaComplex2 = gd.AlphaComplex(points=pointCloud2)
    st_alpha2 = alphaComplex2.create_simplex_tree()

    print("alpha done")

    print("Constructed Simplicial Complexes")
    # Pre vs. Post GH Distances

    vertices = [simplex for simplex, _ in st_rips1.get_filtration() if len(simplex) == 1]
    edges = [simplex for simplex, _ in st_rips1.get_filtration() if len(simplex) == 2]

    repr1 = nhg.represent_simplicial_complex(vertices, )

    geo1 = nhg.geometric_realization_point_cloud(st_rips1)
    geo2 = nhg.geometric_realization_point_cloud(st_rips2)

    print("got geo relizations")

    hausdorff_distance = nhg.compute_hausdorff_distance(geo1, geo2)
    #gromov_hausdorff_distance = nhg.compute_gromov_hausdorff_distance(geo1, geo2)

    print(hausdorff_distance) 

    


if __name__ == "__main__":
    dmitrii_kashtanov_konstantinovich()
