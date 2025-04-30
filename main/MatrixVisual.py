import os
import platform
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def _open(path):
    if platform.system() == 'Darwin':
        subprocess.run(['open', path])
    elif platform.system() == 'Windows':
        os.startfile(path)
    else:
        subprocess.run(['xdg-open', path])


def save_heatmap(csv_path, out_path):
    df = pd.read_csv(csv_path, index_col=0)
    plt.figure(figsize=(16, 14))
    plt.pcolormesh(df, cmap='coolwarm', shading='auto',
                   vmin=-np.max(np.abs(df.values)), vmax=np.max(np.abs(df.values)))
    plt.title('Covariance Heatmap')
    plt.axis('off')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    _open(out_path)


def extract_edges(csv_path, threshold=0.7):
    df = pd.read_csv(csv_path, index_col=0)
    edges = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            cov = df.iat[i, j]
            if abs(cov) >= threshold:
                edges.append((df.index[i], df.columns[j], cov))
    return pd.DataFrame(edges, columns=['source', 'target', 'covariance'])


def draw_network(edge_source, out_path,
                 threshold=0.7, min_width=0.5, max_width=12, accent=2,
                 seed=42, cluster_offset=1.5, k_scale=10.0,
                 show_node_labels=0):

    # Load edges
    edges_df = pd.read_csv(edge_source) if isinstance(edge_source, str) else edge_source
    G = nx.Graph()
    for u, v, cov in edges_df.itertuples(index=False):
        G.add_edge(u, v, weight=abs(cov), cov=cov)

    # Split nodes by color
    blue = [n for n in G if not n.startswith('R_')]
    red = [n for n in G if n.startswith('R_')]

    # Compute spring layouts separately for clusters
    if len(blue) > 1:
        k_blue = k_scale * (1 / np.sqrt(len(blue)))
    else:
        k_blue = None
    if len(red) > 1:
        k_red = k_scale * (1 / np.sqrt(len(red)))
    else:
        k_red = None
    pos_blue = nx.spring_layout(G.subgraph(blue), seed=seed, k=k_blue)
    pos_red = nx.spring_layout(G.subgraph(red), seed=seed, k=k_red)

    # Shift clusters apart by reduced offset
    for n, (x, y) in pos_blue.items():
        pos_blue[n] = (x - cluster_offset, y)
    for n, (x, y) in pos_red.items():
        pos_red[n] = (x + cluster_offset, y)

    # Combine positions
    pos = {**pos_blue, **pos_red}

    # Node appearance
    colors = ['red' if n in red else 'blue' for n in G]
    node_size = 300

    # Edge widths: exponential gradient accentuating differences
    widths = []
    for _, _, data in G.edges(data=True):
        w = data['weight']
        norm = np.clip((w - threshold) / (1.0 - threshold), 0, 1)
        w_scaled = min_width + (norm**accent) * (max_width - min_width)
        widths.append(w_scaled)

    plt.figure(figsize=(16, 14))
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=widths,
        edge_color='gray',
        alpha=0.8
    )
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        node_size=node_size,
        edgecolors='k'
    )
    # Optionally draw node labels
    if show_node_labels:
        nx.draw_networkx_labels(G, pos, font_size=9)

    # Draw edge labels with covariance values
    edge_labels = {(u, v): f"{d['cov']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8
    )

    plt.title('Brain Network')
    plt.axis('off')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    _open(out_path)
    return G


if __name__ == '__main__':
    cov_csv = '/Users/liamm/Documents/mchacks/CookLab/.venv/data/Post_0_covariance.csv'
    edges_csv = 'data/edges.csv'
    os.makedirs(os.path.dirname(cov_csv), exist_ok=True)
    os.makedirs(os.path.dirname(edges_csv), exist_ok=True)

    edges_df = extract_edges(cov_csv)
    edges_df.to_csv(edges_csv, index=False)

    save_heatmap(cov_csv, 'visuals/heatmap.png')
    draw_network(edges_csv, 'visuals/network_graph.png', show_node_labels=1)
