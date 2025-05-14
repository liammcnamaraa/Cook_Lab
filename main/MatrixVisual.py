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


def visualize_communities_3d(supra_adj, communities, region_labels, unique_regions, output_file="CommunityMap0.png"):
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

def visualize_network_parameters(supra_adj, communities, region_labels, unique_regions, output_file="ParameterSummary0.png", save_figure=True):
    """
    Visualize the four key network parameters described in Standage et al. (2024):
    - Disjointedness: independent module changes
    - Cohesion Strength: coordinated module changes
    - Recruitment: consistency of within-network interactions
    - Integration: normalized between-network interactions
    
    Parameters:
    -----------
    supra_adj : array-like
        The supra-adjacency matrix connecting pre and post states
    communities : dict
        Mapping of node indices to community IDs
    region_labels : list
        List of region labels
    unique_regions : list
        List of unique brain regions
    output_file : str, optional
        Output file path for the visualization
    
    Returns:
    --------
    fig : matplotlib figure
        The complete visualization figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    # Convert supra_adj to numpy array if sparse
    if sparse.issparse(supra_adj):
        supra_adj = supra_adj.toarray()
    
    # Get number of regions
    n_regions = len(unique_regions)
    
    # Setup figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig)
    
    # ---- 1. Create module assignments matrix (T, N) as in paper ----
    # Where T=2 time windows (pre and post) and N=number of regions
    module_assignments = np.zeros((2, n_regions), dtype=int)
    for i in range(n_regions):
        module_assignments[0, i] = communities[i]  # pre
        module_assignments[1, i] = communities[i + n_regions]  # post
    
    # ---- 2. Compute parameters using formulas from the paper ----
    # Compute disjointedness - measures independent module changes
    disjointedness = compute_disjointedness(module_assignments)
    
    # Compute cohesion strength - measures coordinated module reconfiguration
    cohesion = compute_cohesion_strength(module_assignments)
    
    # Create module allegiance matrix (P) as described in the paper
    # P_ij = probability that regions i and j were in same module
    allegiance = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            # Check both time points
            same_module_count = 0
            for t in range(2):  # pre and post
                i_module = communities[i] if t == 0 else communities[i + n_regions]
                j_module = communities[j] if t == 0 else communities[j + n_regions]
                if i_module == j_module:
                    same_module_count += 1
            allegiance[i, j] = same_module_count / 2
    
    # Create network labels from community assignments
    # In the paper, networks are derived by clustering the module allegiance matrix
    # For simplicity, we'll use the first time point community assignments 
    network_labels = np.array([communities[i] for i in range(n_regions)])
    
    # Compute recruitment - consistency of within-network interactions
    recruitment = compute_recruitment(allegiance, network_labels)
    
    # Compute integration - normalized between-network interactions
    integration = compute_integration(allegiance, network_labels)
    
    # ---- 3. Visualize the parameters ----
    
    # 3.1. Disjointedness and Cohesion Strength Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    param_df = pd.DataFrame({
        'Region': unique_regions,
        'Disjointedness': disjointedness,
        'Cohesion Strength': cohesion
    })
    
    # Sort by disjointedness for better visualization
    param_df = param_df.sort_values('Disjointedness', ascending=False)
    
    # Reshape for plotting with seaborn
    param_df_melt = pd.melt(
        param_df, 
        id_vars=['Region'], 
        value_vars=['Disjointedness', 'Cohesion Strength']
    )
    
    # Plot the comparison
    sns.barplot(x='Region', y='value', hue='variable', data=param_df_melt, ax=ax1)
    ax1.set_title('Disjointedness vs. Cohesion Strength by Region', fontsize=14)
    ax1.set_xlabel('Brain Regions', fontsize=12)
    ax1.set_ylabel('Parameter Value', fontsize=12)
    ax1.tick_params(axis='x', rotation=90)
    ax1.legend(title='Parameter')
    
    # Show only a subset of regions if there are many
    if len(unique_regions) > 20:
        ax1.set_xticks(ax1.get_xticks()[::len(unique_regions)//20])
    
    # 3.2. Module Transitions Visualization
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # Create transition dataframe
    transitions = []
    for i in range(n_regions):
        pre_module = communities[i]
        post_module = communities[i + n_regions]
        transitions.append({
            'Region': unique_regions[i],
            'Pre Module': f'Module {pre_module}',
            'Post Module': f'Module {post_module}'
        })
    
    trans_df = pd.DataFrame(transitions)
    
    # Count transitions between modules
    transition_counts = trans_df.groupby(['Pre Module', 'Post Module']).size().reset_index(name='count')
    
    # Get unique modules
    pre_modules = sorted(trans_df['Pre Module'].unique())
    post_modules = sorted(trans_df['Post Module'].unique())
    
    # Create transition matrix
    trans_matrix = np.zeros((len(pre_modules), len(post_modules)))
    for _, row in transition_counts.iterrows():
        i = pre_modules.index(row['Pre Module'])
        j = post_modules.index(row['Post Module'])
        trans_matrix[i, j] = row['count']
    
    # Plot transition heatmap
    sns.heatmap(trans_matrix, annot=True, fmt='g', cmap='viridis',
                xticklabels=post_modules, yticklabels=pre_modules, ax=ax2)
    ax2.set_title('Module Transitions (Pre → Post)', fontsize=14)
    ax2.set_xlabel('Post Modules', fontsize=12)
    ax2.set_ylabel('Pre Modules', fontsize=12)
    
    # 3.3. Recruitment by Network
    ax3 = fig.add_subplot(gs[1, :2])
    
    # Convert recruitment dict to DataFrame
    recruitment_df = pd.DataFrame({
        'Network': [f'Network {k}' for k in recruitment.keys()],
        'Recruitment': list(recruitment.values())
    })
    
    # Plot recruitment
    sns.barplot(x='Network', y='Recruitment', data=recruitment_df, palette='Set2', ax=ax3)
    ax3.set_title('Recruitment by Network', fontsize=14)
    ax3.set_xlabel('Network', fontsize=12)
    ax3.set_ylabel('Recruitment Score (Ik,k)', fontsize=12)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax3.text(len(recruitment_df)-0.5, 0.52, 'Threshold', color='red')
    
    # 3.4. Integration Between Networks
    ax4 = fig.add_subplot(gs[1, 2:])
    
    # Convert integration dict to matrix
    k_max = max(max(k) for k in integration.keys()) + 1
    int_matrix = np.zeros((k_max, k_max))
    
    # Fill integration matrix
    for (k1, k2), value in integration.items():
        int_matrix[k1, k2] = value
        int_matrix[k2, k1] = value  # Symmetric
    
    # Fill diagonal with 1s (self-integration)
    np.fill_diagonal(int_matrix, 1.0)
    
    # Network labels
    network_names = [f'Network {i}' for i in range(k_max)]
    
    # Plot integration matrix (I'k1,k2)
    sns.heatmap(int_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=network_names, yticklabels=network_names, ax=ax4,
                vmin=0, vmax=1)
    ax4.set_title('Integration Between Networks (I\'k1,k2)', fontsize=14)
    ax4.set_xlabel('Network', fontsize=12)
    ax4.set_ylabel('Network', fontsize=12)
    
    # 3.5. Combined Parameter Summary
    ax5 = fig.add_subplot(gs[2, :2])
    
    # Calculate mean values for each parameter
    mean_disjoint = np.mean(disjointedness)
    mean_cohesion = np.mean(cohesion)
    mean_recruitment = np.nanmean(list(recruitment.values()))
    mean_integration = np.nanmean(list(integration.values()))
    
    # Count regions that changed modules
    changes = sum(1 for i in range(n_regions) 
                  if communities[i] != communities[i + n_regions])
    stability = 1 - (changes / n_regions)
    
    # Create summary metrics
    summary_df = pd.DataFrame({
        'Parameter': ['Disjointedness', 'Cohesion Strength', 'Mean Recruitment', 'Mean Integration', 'Network Stability'],
        'Value': [mean_disjoint, mean_cohesion, mean_recruitment, mean_integration, stability]
    })
    
    # Plot summary metrics
    sns.barplot(x='Parameter', y='Value', data=summary_df, palette='Set3', ax=ax5)
    ax5.set_title('Parameter Summary', fontsize=14)
    ax5.set_xlabel('Parameter', fontsize=12)
    ax5.set_ylabel('Mean Value', fontsize=12)
    ax5.tick_params(axis='x', rotation=45)
    
    # 3.6. Network Statistics
    ax6 = fig.add_subplot(gs[2, 2:])
    
    # Create summary text
    summary_text = (
        f"Network Statistics Summary\n"
        f"==========================\n\n"
        f"Total Regions: {n_regions}\n"
        f"Regions Changed Modules: {changes} ({changes/n_regions:.1%})\n"
        f"Network Stability: {stability:.1%}\n\n"
        f"Number of Communities: {len(set(communities.values()))}\n\n"
        f"Disjointedness: {mean_disjoint:.3f}\n"
        f"Cohesion Strength: {mean_cohesion:.3f}\n"
        f"Mean Recruitment: {mean_recruitment:.3f}\n"
        f"Mean Integration: {mean_integration:.3f}\n\n"
        f"These parameters represent the network dynamics as described in Standage et al. (2024):\n"
        f"- Disjointedness: Degree to which regions change modules independently\n"
        f"- Cohesion Strength: Degree to which regions change modules together\n"
        f"- Recruitment: Consistency of within-network interactions (I_k,k)\n"
        f"- Integration: Normalized between-network interactions (I'_k1,k2)\n"
    )
    
    # Remove plot elements
    ax6.axis('off')
    
    # Add descriptive text
    ax6.text(0, 0.5, summary_text, fontsize=12, va='center',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    # Add tight layout
    plt.tight_layout()
    
    # Save figure
    if save_figure:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Network parameter visualization saved to {output_file}")
    
    return fig

def analyze_brain_networks(pre_cov_file, post_cov_file, 
                         output_dir="brain_network_analysis",
                         threshold=0.7, omega=1.0, 
                         resolution=1.0, random_state=42):
    """
    Complete analysis pipeline for brain network parameters starting from covariance matrices.
    
    Parameters:
    -----------
    pre_cov_file : str
        Path to the pre-condition covariance matrix CSV
    post_cov_file : str
        Path to the post-condition covariance matrix CSV
    output_dir : str, optional
        Directory to save output files
    threshold : float, optional
        Threshold for converting covariance to edges (default=0.7)
    omega : float, optional
        Inter-slice coupling parameter (default=1.0)
    resolution : float, optional
        Resolution parameter (gamma) for community detection
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict : 
        Dictionary containing results and file paths
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths
    pre_edges_file = os.path.join(output_dir, "pre_edges.csv")
    post_edges_file = os.path.join(output_dir, "post_edges.csv")
    supra_viz_path = os.path.join(output_dir, "supra_adjacency_viz.png")
    community_viz_path = os.path.join(output_dir, "community_3d_viz.png")
    param_viz_path = os.path.join(output_dir, "network_parameters_summary.png")
    
    # 1. Convert covariance matrices to edge lists
    print("Converting covariance matrices to edge lists...")
    
    # For pre-condition
    pre_df = pd.read_csv(pre_cov_file, index_col=0)
    pre_edges = []
    for i in range(len(pre_df)):
        for j in range(i + 1, len(pre_df)):  # Upper triangle to avoid duplicates
            cov = pre_df.iloc[i, j]
            if abs(cov) >= threshold:
                pre_edges.append((pre_df.index[i], pre_df.columns[j], cov))
    
    pre_edges_df = pd.DataFrame(pre_edges, columns=['source', 'target', 'covariance'])
    pre_edges_df.to_csv(pre_edges_file, index=False)
    
    # For post-condition
    post_df = pd.read_csv(post_cov_file, index_col=0)
    post_edges = []
    for i in range(len(post_df)):
        for j in range(i + 1, len(post_df)):
            cov = post_df.iloc[i, j]
            if abs(cov) >= threshold:
                post_edges.append((post_df.index[i], post_df.columns[j], cov))
    
    post_edges_df = pd.DataFrame(post_edges, columns=['source', 'target', 'covariance'])
    post_edges_df.to_csv(post_edges_file, index=False)
    
    # 2. Create adjacency matrices from edge lists
    print("Creating adjacency matrices...")
    
    # Get all unique regions
    pre_regions = list(set(pre_edges_df['source'].tolist() + pre_edges_df['target'].tolist()))
    post_regions = list(set(post_edges_df['source'].tolist() + post_edges_df['target'].tolist()))
    all_regions = sorted(list(set(pre_regions + post_regions)))
    n_regions = len(all_regions)
    
    # Create pre adjacency matrix
    pre_adj = pd.DataFrame(0.0, index=all_regions, columns=all_regions)
    for _, row in pre_edges_df.iterrows():
        pre_adj.loc[row['source'], row['target']] = row['covariance']
        pre_adj.loc[row['target'], row['source']] = row['covariance']  # Symmetric
    
    # Create post adjacency matrix
    post_adj = pd.DataFrame(0.0, index=all_regions, columns=all_regions)
    for _, row in post_edges_df.iterrows():
        post_adj.loc[row['source'], row['target']] = row['covariance']
        post_adj.loc[row['target'], row['source']] = row['covariance']  # Symmetric
    
    # Save adjacency matrices
    pre_adj_file = os.path.join(output_dir, "pre_adjacency.csv")
    post_adj_file = os.path.join(output_dir, "post_adjacency.csv")
    pre_adj.to_csv(pre_adj_file)
    post_adj.to_csv(post_adj_file)
    
    # 3. Create supra-adjacency matrix
    print("Creating supra-adjacency matrix...")
    supra_adj, region_labels, unique_regions, n_regions = create_supra_adjacency_matrix(
        pre_adj_file, post_adj_file, omega=omega
    )
    
    # 5. Detect communities using Louvain algorithm
    print("Detecting communities...")
    communities, Q = detect_communities_louvain(supra_adj, gamma=resolution, random_state=random_state)
    print(f"Detected {len(set(communities.values()))} communities")
    print(f"Modularity score (Q): {Q:.4f}")
    
    # 6. Visualize communities in 3D
    print("Visualizing communities in 3D...")
    visualize_communities_3d(supra_adj, communities, region_labels, unique_regions, 
                           output_file=community_viz_path)
    
    # 7. Calculate and visualize network parameters
    print("Calculating and visualizing network parameters...")
    param_fig = visualize_network_parameters(
        supra_adj, communities, region_labels, unique_regions, output_file=param_viz_path
    )
    
    # 8. Create module allegiance matrix and analyze communities
    print("Analyzing module allegiance and community structure...")
    
    # Create module assignments matrix (T=2 time points, N=n_regions)
    module_assignments = np.zeros((2, n_regions), dtype=int)
    for i in range(n_regions):
        module_assignments[0, i] = communities[i]  # pre
        module_assignments[1, i] = communities[i + n_regions]  # post
    
    # Calculate disjointedness and cohesion strength
    disjointedness = compute_disjointedness(module_assignments)
    cohesion = compute_cohesion_strength(module_assignments)
    
    # Calculate module allegiance matrix
    allegiance = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            # Check both time points
            same_module_count = 0
            for t in range(2):  # pre and post
                i_module = communities[i] if t == 0 else communities[i + n_regions]
                j_module = communities[j] if t == 0 else communities[j + n_regions]
                if i_module == j_module:
                    same_module_count += 1
            allegiance[i, j] = same_module_count / 2
    
    # Visualization of module allegiance matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(allegiance, cmap='viridis', interpolation='none')
    plt.colorbar(label='Probability of same module')
    plt.title('Module Allegiance Matrix')
    plt.xlabel('Region Index')
    plt.ylabel('Region Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "module_allegiance.png"), dpi=300, bbox_inches='tight')
    
    # Identify top regions by cohesion and disjointedness
    top_cohesive_idx = np.argsort(cohesion)[-5:][::-1]  # Top 5 regions
    top_disjointed_idx = np.argsort(disjointedness)[-5:][::-1]  # Top 5 regions
    
    # Create and save a summary report
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("BRAIN NETWORK ANALYSIS SUMMARY\n")
        f.write("==============================\n\n")
        f.write(f"Number of regions analyzed: {n_regions}\n")
        f.write(f"Number of communities detected: {len(set(communities.values()))}\n")
        f.write(f"Modularity score (Q): {Q:.4f}\n\n")
        
        f.write("Network Parameters:\n")
        f.write(f"- Mean Disjointedness: {np.mean(disjointedness):.4f}\n")
        f.write(f"- Mean Cohesion Strength: {np.mean(cohesion):.4f}\n\n")
        
        f.write("Top 5 regions with highest cohesion strength:\n")
        for idx in top_cohesive_idx:
            f.write(f"- Region {unique_regions[idx]}: {cohesion[idx]:.4f}\n")
        
        f.write("\nTop 5 regions with highest disjointedness:\n")
        for idx in top_disjointed_idx:
            f.write(f"- Region {unique_regions[idx]}: {disjointedness[idx]:.4f}\n")
    
    print(f"Analysis complete! Results saved to {output_dir}")
    
    # Return results dictionary
    results = {
        "supra_adjacency_matrix": supra_adj,
        "region_labels": region_labels,
        "communities": communities,
        "modularity_score": Q,
        "disjointedness": disjointedness,
        "cohesion_strength": cohesion,
        "module_allegiance": allegiance,
        "visualizations": {
            "supra_adjacency": supra_viz_path,
            "communities_3d": community_viz_path,
            "parameters": param_viz_path,
            "module_allegiance": os.path.join(output_dir, "module_allegiance.png")
        },
        "summary": summary_file
    }
    
    return results

if __name__ == '__main__':
    pre_file = "/Users/liamm/Documents/Cook_Lab/data/joe_data/Pre_3_covariance.csv"
    post_file = "/Users/liamm/Documents/Cook_Lab/data/joe_data/Post_3_covariance.csv"
    
    # Run the complete brain network analysis starting directly from covariance matrices
    results = analyze_brain_networks(
        pre_file, 
        post_file, 
        output_dir="analysis3",
        threshold=0.7,  # Threshold for converting covariance to edges
        omega=1.0,      # Inter-slice coupling parameter
        resolution=1.0, # Resolution parameter for community detection
        random_state=42 # For reproducibility
    )