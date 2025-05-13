import os
import platform
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import networkx as nx
from scipy import sparse
import seaborn as sns


def _open(path):
    """Open a file with the default application based on OS."""
    if platform.system() == 'Darwin':
        subprocess.run(['open', path])
    elif platform.system() == 'Windows':
        os.startfile(path)
    else:
        subprocess.run(['xdg-open', path])


def save_heatmap(csv_path, out_path):
    """Create and save a heatmap visualization of a covariance matrix."""
    df = pd.read_csv(csv_path, index_col=0)
    plt.figure(figsize=(16, 14))
    max_abs = np.max(np.abs(df.values))
    plt.pcolormesh(df, cmap='coolwarm', shading='auto', vmin=-max_abs, vmax=max_abs)
    plt.title('Covariance Heatmap')
    plt.axis('off')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    _open(out_path)


def extract_edges(csv_path, threshold=0.7):
    """Extract edges from a covariance matrix based on threshold."""
    df = pd.read_csv(csv_path, index_col=0)
    edges = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            cov = df.iat[i, j]
            if abs(cov) >= threshold:
                edges.append((df.index[i], df.columns[j], 1))
    return pd.DataFrame(edges, columns=['source', 'target', 'covariance'])


def draw_network(edge_source, out_path, threshold=0.7, min_width=0.5, max_width=12, 
                accent=2, seed=42, cluster_offset=1.5, k_scale=10.0, show_node_labels=""):
    """Draw a network visualization based on edge data."""
    # Load edges
    edges_df = pd.read_csv(edge_source) if isinstance(edge_source, str) else edge_source
    G = nx.Graph()
    for u, v, cov in edges_df.itertuples(index=False):
        G.add_edge(u, v, weight=abs(cov), cov=cov)

    # Split nodes by hemisphere (left/right)
    blue = [n for n in G if not n.startswith('R_')]
    red = [n for n in G if n.startswith('R_')]

    # Compute spring layouts for each hemisphere
    k_blue = k_scale * (1 / np.sqrt(len(blue))) if len(blue) > 1 else None
    k_red = k_scale * (1 / np.sqrt(len(red))) if len(red) > 1 else None
    
    pos_blue = nx.spring_layout(G.subgraph(blue), seed=seed, k=k_blue)
    pos_red = nx.spring_layout(G.subgraph(red), seed=seed, k=k_red)

    # Shift hemispheres apart
    for n, (x, y) in pos_blue.items():
        pos_blue[n] = (x - cluster_offset, y)
    for n, (x, y) in pos_red.items():
        pos_red[n] = (x + cluster_offset, y)

    # Combine positions
    pos = {**pos_blue, **pos_red}

    # Calculate edge widths
    widths = []
    for _, _, data in G.edges(data=True):
        w = data['weight']
        norm = np.clip((w - threshold) / (1.0 - threshold), 0, 1)
        w_scaled = min_width + (norm**accent) * (max_width - min_width)
        widths.append(w_scaled)

    # Create visualization
    plt.figure(figsize=(16, 14))
    nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray', alpha=0.8)
    nx.draw_networkx_nodes(G, pos, node_color=['red' if n in red else 'blue' for n in G],
                          node_size=300, edgecolors='k')
    
    # Draw edge labels with covariance values
    edge_labels = {(u, v): f"{d['cov']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title('Brain Network')
    plt.axis('off')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    _open(out_path)
    return G


def edges_to_adjacency_matrix(edges_file):
    """Convert edge list to adjacency matrix."""
    edges_df = pd.read_csv(edges_file)
    
    # Get all unique nodes
    sources = edges_df['source'].unique()
    targets = edges_df['target'].unique()
    nodes = sorted(list(set(list(sources) + list(targets))))
    
    # Create and fill adjacency matrix
    n = len(nodes)
    adj_matrix = pd.DataFrame(np.zeros((n, n)), index=nodes, columns=nodes)
    
    for _, row in edges_df.iterrows():
        source, target, covariance = row['source'], row['target'], row['covariance']
        adj_matrix.loc[source, target] = covariance
        adj_matrix.loc[target, source] = covariance
    
    return adj_matrix, nodes


def create_supra_adjacency_matrix(pre_adj_file, post_adj_file, omega=1.0):
    """Create a supra-adjacency matrix connecting pre and post adjacency matrices."""
    # Read adjacency matrices
    pre_adj = pd.read_csv(pre_adj_file, index_col=0)
    post_adj = pd.read_csv(post_adj_file, index_col=0)
    
    # Get all unique regions
    pre_regions = list(pre_adj.index)
    post_regions = list(post_adj.index)
    all_regions = sorted(list(set(pre_regions + post_regions)))
    n_regions = len(all_regions)
    
    # Create expanded adjacency matrices with all regions
    pre_expanded = pd.DataFrame(0.0, index=all_regions, columns=all_regions)
    post_expanded = pd.DataFrame(0.0, index=all_regions, columns=all_regions)
    
    # Fill in existing connections
    for i, row_region in enumerate(pre_regions):
        for j, col_region in enumerate(pre_regions):
            pre_expanded.loc[row_region, col_region] = pre_adj.loc[row_region, col_region]
            
    for i, row_region in enumerate(post_regions):
        for j, col_region in enumerate(post_regions):
            post_expanded.loc[row_region, col_region] = post_adj.loc[row_region, col_region]
    
    # Convert to numpy arrays
    pre_np = pre_expanded.values
    post_np = post_expanded.values
    
    # Create block diagonal matrix with both time slices
    intra_slice = sparse.block_diag([pre_np, post_np])
    
    # Create inter-slice connections
    inter_slice_entries = []
    for i, region in enumerate(all_regions):
        row_idx = i
        col_idx = i + n_regions
        inter_slice_entries.append((row_idx, col_idx, omega))
        inter_slice_entries.append((col_idx, row_idx, omega))
    
    rows, cols, data = zip(*inter_slice_entries)
    inter_slice = sparse.csr_matrix(
        (data, (rows, cols)), 
        shape=(2*n_regions, 2*n_regions)
    )
    
    # Combine intra- and inter-slice connections
    supra_adj = intra_slice + inter_slice
    
    # Create region labels
    region_labels = []
    for t in ["pre_", "post_"]:
        for r in all_regions:
            region_labels.append(f"{t}{r}")
    
    return supra_adj, region_labels, all_regions, n_regions


def visualize_supra_adjacency(pre_adj_file, post_adj_file, omega=1.0, output_file="supra_adjacency_viz.png"):
    """Visualize a supra-adjacency matrix connecting pre and post matrices."""
    # Create the supra-adjacency matrix
    supra_adj, region_labels, all_regions, n_regions = create_supra_adjacency_matrix(
        pre_adj_file, post_adj_file, omega
    )
    
    # Convert to dense array for visualization
    supra_dense = supra_adj.toarray()
    
    # Create overview visualization
    plt.figure(figsize=(16, 14))
    cmap = plt.cm.coolwarm
    
    # Set color normalization
    vmax = max(
        np.abs(supra_dense[:n_regions, :n_regions]).max(),
        np.abs(supra_dense[n_regions:, n_regions:]).max()
    )
    
    # Plot matrix
    im = plt.imshow(supra_dense, cmap=cmap, vmin=-vmax, vmax=vmax)
    
    # Add colorbar and dividing lines
    cbar = plt.colorbar(im)
    cbar.set_label('Connection Strength')
    plt.axhline(y=n_regions-0.5, color='black', linestyle='-', linewidth=2)
    plt.axvline(x=n_regions-0.5, color='black', linestyle='-', linewidth=2)
    
    # Add quadrant labels
    plt.text(n_regions/4, n_regions/4, f"Pre-Pre\n({n_regions}×{n_regions})", 
             ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.text(n_regions + n_regions/4, n_regions/4, f"Pre-Post\n(Inter-slice, ω={omega})", 
             ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.text(n_regions/4, n_regions + n_regions/4, f"Post-Pre\n(Inter-slice, ω={omega})", 
             ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.text(n_regions + n_regions/4, n_regions + n_regions/4, f"Post-Post\n({n_regions}×{n_regions})", 
             ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add title and labels
    plt.title('Supra-Adjacency Matrix (Pre + Post)', fontsize=16)
    plt.xlabel('Node Index', fontsize=14)
    plt.ylabel('Node Index', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Create detailed visualization
    plt.figure(figsize=(20, 18))
    with sns.axes_style("white"):
        sns.heatmap(supra_dense, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                   square=True, linewidths=0, cbar_kws={"shrink": 0.8})
    
    plt.title('Detailed Supra-Adjacency Matrix with Inter-Slice Connections', fontsize=16)
    plt.text(-0.05, n_regions/2, 'Pre', fontsize=14, ha='center', va='center', rotation=90)
    plt.text(-0.05, n_regions + n_regions/2, 'Post', fontsize=14, ha='center', va='center', rotation=90)
    plt.text(n_regions/2, -0.05, 'Pre', fontsize=14, ha='center', va='center')
    plt.text(n_regions + n_regions/2, -0.05, 'Post', fontsize=14, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig("detailed_" + output_file, dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {output_file} and detailed_{output_file}")
    print(f"Matrix shape: {supra_dense.shape}")
    print(f"Number of unique regions: {n_regions}")
    
    return supra_dense, region_labels


def detect_communities_louvain(supra_adj, gamma=1.0, random_state=None):
    """Detect communities in a supra-adjacency matrix using Louvain algorithm."""
    if sparse.issparse(supra_adj):
        supra_adj = supra_adj.toarray()
    
    # Create weighted graph
    G = nx.from_numpy_array(supra_adj)
    
    # Apply Louvain algorithm
    communities = nx.community.louvain_communities(G, resolution=gamma, seed=random_state)
    
    # Convert to dict mapping nodes to community IDs
    community_dict = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            community_dict[node] = community_id
    
    # Calculate modularity score
    Q = nx.community.modularity(G, communities)
    
    return community_dict, Q


def visualize_communities_3d(supra_adj, communities, region_labels, unique_regions, output_file="community_3d_viz.png"):
    """Visualize communities in a 3D graph with time slices."""
    # Convert to numpy array if sparse
    if sparse.issparse(supra_adj):
        supra_adj = supra_adj.toarray()
    
    # Number of regions and time slices
    n_regions = len(unique_regions)
    n_time_slices = 2  # pre and post
    
    # Create a graph
    G = nx.from_numpy_array(supra_adj)
    
    # Generate a colormap for communities
    n_communities = len(set(communities.values()))
    colors = plt.cm.tab20(np.linspace(0, 1, n_communities))
    
    # Create a 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate positions for nodes in 3D space
    pos = {}
    for time_slice in range(n_time_slices):
        # Extract subgraph for this time slice
        slice_nodes = list(range(time_slice * n_regions, (time_slice + 1) * n_regions))
        subgraph = G.subgraph(slice_nodes)
        
        # Get 2D layout for this slice
        slice_pos = nx.spring_layout(subgraph, seed=42)
        
        # Add z-coordinate for time slice
        z_coord = time_slice * 5  # Separation between time slices
        
        # Add positions to the full position dictionary
        for node, (x, y) in slice_pos.items():
            pos[node] = (x, y, z_coord)
    
    # Draw edges between time slices (inter-slice connections)
    for i in range(n_regions):
        node1 = i
        node2 = i + n_regions
        if G.has_edge(node1, node2):
            x1, y1, z1 = pos[node1]
            x2, y2, z2 = pos[node2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', alpha=0.2, linewidth=0.5)
    
    # Draw edges within each time slice (intra-slice connections)
    for time_slice in range(n_time_slices):
        start_idx = time_slice * n_regions
        end_idx = (time_slice + 1) * n_regions
        
        for u, v in G.edges():
            if start_idx <= u < end_idx and start_idx <= v < end_idx:
                x1, y1, z1 = pos[u]
                x2, y2, z2 = pos[v]
                
                # Get community colors
                u_community = communities[u]
                v_community = communities[v]
                
                # If nodes are in the same community, draw the edge
                if u_community == v_community:
                    edge_color = colors[u_community]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color=edge_color, alpha=0.6, linewidth=1)
    
    # Draw nodes
    for node, (x, y, z) in pos.items():
        community_id = communities[node]
        color = colors[community_id]
        ax.scatter(x, y, z, color=color, s=100, edgecolors='k', alpha=0.8)
    
    # Add time slice labels
    ax.text(0, 0, 0, "Pre", fontsize=14, horizontalalignment='center')
    ax.text(0, 0, 5, "Post", fontsize=14, horizontalalignment='center')
    
    # Set plot limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time Slice')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Add title and legend
    plt.title(f'3D Visualization of {n_communities} Communities across Time Slices', fontsize=16)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], markersize=10, 
                                 label=f'Community {i}') for i in range(n_communities)]
    ax.legend(handles=legend_elements, loc='upper right', title="Communities")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"3D community visualization saved to {output_file}")
    return fig, ax

def compute_disjointedness(module_assignments):
    """
    Compute disjointedness for each node.
    module_assignments: int array of shape (T, N) giving module indices per time window
    Returns: disjointedness: float array of length N
    """
    T, N = module_assignments.shape
    transitions = T - 1
    # count independent changes per node
    disjoint_counts = np.zeros(N)
    for t in range(transitions):
        before = module_assignments[t]
        after = module_assignments[t+1]
        for i in range(N):
            if before[i] != after[i]:
                # check if any other node changed with it
                others = [j for j in range(N) if j != i]
                joint = any((before[j] != after[j]) and (after[j] == after[i]) for j in others)
                # if no joint movement, it's disjoint
                if not joint:
                    disjoint_counts[i] += 1
    return disjoint_counts / transitions


def compute_cohesion_strength(module_assignments):
    """
    Compute cohesion strength (coordinated reconfiguration) for each node.
    Returns: cohesion_strength: float array of length N
    """
    T, N = module_assignments.shape
    transitions = T - 1
    coh_counts = np.zeros(N)
    for t in range(transitions):
        before = module_assignments[t]
        after = module_assignments[t+1]
        # for each pair of nodes, if both change into same new module, count for each
        for i in range(N):
            for j in range(i+1, N):
                if before[i] != after[i] and before[j] != after[j] and after[i] == after[j]:
                    coh_counts[i] += 1
                    coh_counts[j] += 1
    # normalize by transitions
    return coh_counts / transitions


def compute_recruitment(allegiance, network_labels):
    """
    Compute recruitment for each network.
    allegiance: numpy array shape (N, N), P_ij probability nodes i and j in same module
    network_labels: array of length N with labels 0..K-1 indicating network membership
    Returns: recruitment: dict mapping network k to recruitment value
    """
    K = network_labels.max() + 1
    recruitment = {}
    for k in range(K):
        idx = np.where(network_labels == k)[0]
        if len(idx) == 0:
            recruitment[k] = np.nan
            continue
        # sum over all pairs including self
        sub = allegiance[np.ix_(idx, idx)]
        recruitment[k] = sub.sum() / (len(idx)**2)
    return recruitment


def compute_integration(allegiance, network_labels):
    """
    Compute integration between all distinct network pairs.
    Returns: integration: dict mapping (k1,k2) to integration value
    """
    K = network_labels.max() + 1
    # first compute raw interactions
    I = np.zeros((K, K))
    for k1 in range(K):
        idx1 = np.where(network_labels == k1)[0]
        for k2 in range(K):
            idx2 = np.where(network_labels == k2)[0]
            if len(idx1)==0 or len(idx2)==0:
                I[k1, k2] = np.nan
            else:
                sub = allegiance[np.ix_(idx1, idx2)]
                I[k1, k2] = sub.sum() / (len(idx1)*len(idx2))
    # recruitment is diag(I)
    # integration normalized
    integration = {}
    for k1 in range(K):
        for k2 in range(K):
            if k1 != k2:
                integration[(k1, k2)] = I[k1, k2] / np.sqrt(I[k1, k1] * I[k2, k2])
    return integration



if __name__ == '__main__':
    pre_file = "/Users/liamm/Documents/Cook_Lab/data/pre0adjacency.csv"
    post_file = "/Users/liamm/Documents/Cook_Lab/data/post0adjacency.csv"
    
    # Create supra-adjacency matrix
    supra_adj, region_labels, unique_regions, n_regions = create_supra_adjacency_matrix(pre_file, post_file)

    # Detect communities
    communities, Q = detect_communities_louvain(supra_adj)
    
    print(f"Detected {len(set(communities.values()))} communities")
    print(f"Modularity score: {Q:.4f}")   

    # Visualize communities
    visualize_communities_3d(supra_adj, communities, region_labels, unique_regions)