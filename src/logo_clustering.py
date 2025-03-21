import os
from pathlib import Path
import shutil
import sqlite3
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import logging
import logging.config
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from sklearn.metrics.pairwise import pairwise_distances
from hdbscan import HDBSCAN
from tqdm import tqdm

from src.feature_extractor import load_features_and_labels
from src.constants import *

def cluster_logos():
    features, logo_files = load_features_and_labels()

    features = normalize(features, norm='l2')

    # Reduce to 50 dimensions
    pca = PCA(n_components=50)
    features = pca.fit_transform(features)

    # Create a logger
    logger = logging.getLogger('info')

    # --- Perform clustering with DBSCAN ---
    dbscan = DBSCAN(eps=0.005, min_samples=3, metric='cosine')
    labels = dbscan.fit_predict(features)

    # Group logos by clusters
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(logo_files[idx])

    # Log clusters (print to console)
    for cluster_id, logos in clusters.items():
        logger.info(f"Cluster {cluster_id}:")
        for logo in logos:
            logger.info(f"  - {logo}")
        logger.info("")

    display_clustered_points(features, labels, logo_files)

# Cluster logos with HDBSCAN
def cluster_logos_h(saving_clusters=False):
    features, logo_files = load_features_and_labels()

    # Create a logger
    logger = logging.getLogger('info')

    # 1. Feature Preprocessing
    features = normalize(features, norm='l2')
    pca = PCA(n_components=50)
    features = pca.fit_transform(features)

    distance_matrix1 = pairwise_distances(features, metric='cosine').astype(np.float64)
    
    # Use HDBSCAN which handles variable density clusters
    clusterer1 = HDBSCAN(metric='precomputed', 
                      min_cluster_size=5,
                      cluster_selection_epsilon=0.001,
                      core_dist_n_jobs=-1,)
    labels1 = clusterer1.fit_predict(distance_matrix1)

    # Create sub-distance matrix for noise points
    final_labels = labels1.copy()
    noise_mask = labels1 == -1
    noise_indices = np.where(noise_mask)[0]
    distance_matrix2 = distance_matrix1[noise_mask][:, noise_mask]
    
    clusterer2 = HDBSCAN(
        metric='precomputed',
        min_cluster_size=2,
        cluster_selection_epsilon=0.015,
        core_dist_n_jobs=-1
    )
    labels2 = clusterer2.fit_predict(distance_matrix2)

    # Adjust labels2 to avoid ID conflicts
    adjusted_labels = np.where(labels2 != -1, 
                                labels2 + labels1.max() + 1, 
                                -1)
    
    # Update final_labels only for noise points
    final_labels[noise_indices] = adjusted_labels

    # After getting clusters dictionary
    clusters = {}
    for idx, label in enumerate(final_labels):
        domain = os.path.splitext(os.path.basename(logo_files[idx]))[0]
        clusters.setdefault(label, []).append(domain)

    # Log clusters (print to console)
    for cluster_id, logos in clusters.items():
        logger.info(f"Cluster {cluster_id}:")
        for logo in logos:
            logger.info(f"  - {logo}")
        logger.info("")

    # Save to output directories
    if saving_clusters:
        save_clustered_logos(clusters, output_root="clustered_logos")

    save_domains(clusters)
    
    # Visualization and analysis
    display_clustered_points(features, final_labels, logo_files)

# --- Visualize clusters with Plotly ---
def display_clustered_points(features, labels, logo_files):
    # --- Reduce dimensions using t-SNE for 2D visualization ---
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    fig = go.Figure()
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_indices = [i for i, x in enumerate(labels) if x == label]
        x = reduced_features[cluster_indices, 0]
        y = reduced_features[cluster_indices, 1]
        hover_data = [logo_files[i] for i in cluster_indices]
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            name=f"Cluster {label}" if label != -1 else "Noise",
            text=hover_data,
            hoverinfo="text",
            marker=dict(size=10, opacity=0.8)
        ))
    
    fig.update_layout(
        title="Clustered Data Points",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        showlegend=True
    )
    fig.show()

def save_clustered_logos(clusters, output_root="output"):
    # Create output root directory
    os.makedirs(output_root, exist_ok=True)
    
    # Add special directory for noise (cluster -1)
    noise_dir = os.path.join(output_root, "noise")
    os.makedirs(noise_dir, exist_ok=True)

    for cluster_id, file_paths in clusters.items():
        # Determine output directory
        if cluster_id == -1:
            target_dir = noise_dir
        else:
            target_dir = os.path.join(output_root, f"cluster{cluster_id}")
            os.makedirs(target_dir, exist_ok=True)

        # Copy files to cluster directory
        for src_path in tqdm(file_paths, desc=f"Cluster {cluster_id}"):
            if not src_path.lower().endswith('.png'):
                continue

            try:
                # Preserve original filename
                filename = os.path.basename(src_path)
                dest_path = os.path.join(target_dir, filename)
                
                # Copy with metadata preservation
                shutil.copy2(src_path, dest_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")

    print(f"Saved clusters to {output_root}")

def save_domains(clusters):
    # Create SQLite database
    conn = sqlite3.connect(OUTPUT_FILE)
    
    # Find the maximum existing cluster ID (excluding noise)
    existing_ids = [cid for cid in clusters.keys() if cid != -1]
    max_cluster = max(existing_ids) if existing_ids else -1
    
    # Track new IDs for noise domains
    new_id = max_cluster + 1  # Start after the largest existing ID
    
    for cluster_id, domains in clusters.items():
        if cluster_id == -1:
            # Save each noise domain as a separate cluster
            for domain in domains:
                pd.DataFrame({"domain": [domain]}).to_sql(
                    f"cluster_{new_id}",
                    conn,
                    if_exists="replace",
                    index=False
                )
                new_id += 1
        else:
            # Save regular clusters
            pd.DataFrame({"domain": domains}).to_sql(
                f"cluster_{cluster_id}",
                conn,
                if_exists="replace",
                index=False
            )
    
    conn.close()
    print(f"Saved {len(clusters)} clusters (including {new_id - max_cluster -1} noise domains) to clusters.db")
