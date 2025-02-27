import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from langchain_chroma import Chroma
import pandas as pd  # Add pandas import for CSV handling
import matplotlib.colors as mcolors  # For color conversion

# Constants
EDITION = '10th'
NUM_CLUSTERS = 10  # You can adjust this number based on your data

# Load the existing Chroma database without embedding function
persist_directory = f"./data/chroma_rf{EDITION}"
vectorstore = Chroma(
    persist_directory=persist_directory
)

def visualize_embeddings_3d():
    """
    Extract embeddings from Chroma, perform k-means clustering,
    reduce to 3D using PCA, and visualize in a 3D scatter plot.
    """
    print("Loading embeddings from Chroma database...")
    
    # Get all documents and their embeddings - include=["embeddings"] is key!
    documents = vectorstore.get(include=["embeddings", "metadatas", "documents"])
    embeddings = documents['embeddings']
    metadatas = documents['metadatas']
    documents_content = documents['documents']
    
    # print(f"First metadata: {metadatas[0]}")
    print(f"Embeddings type: {type(embeddings)}, shape: {embeddings.shape}")
    
    # Check if embeddings are None or empty using NumPy's size attribute
    if embeddings is None or embeddings.size == 0:
        print("No embeddings found in the database.")
        return
    
    print(f"Found {len(embeddings)} embeddings.")
    print(f"First embedding shape: {embeddings[0].shape}")
    
    # No need to convert to numpy array since it already is one
    embeddings_array = embeddings
    
    # Perform k-means clustering first (before dimensionality reduction)
    print(f"Performing k-means clustering with {NUM_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    clusters = kmeans.fit_predict(embeddings_array)
    
    # Reduce dimensionality to 3D using PCA
    print("Reducing dimensionality to 3D...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings_array)
    
    # Create 3D scatter plot
    print("Creating 3D visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points colored by cluster
    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=clusters,
        cmap='tab10',
        alpha=0.9,
        s=30
    )
    
    # Add labels and title
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(f'3D Visualization of Roadfood {EDITION} Edition Embeddings\nColored by Cluster')
    
    # Add a colorbar for clusters
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster')
    
    # Add hover annotations (works in interactive mode)
    annot = ax.annotate(
        "", 
        xy=(0, 0), 
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    annot.set_visible(False)
    
    def update_annot(ind):
        idx = ind["ind"][0]
        pos = scatter._offsets3d
        annot.xy = (pos[0][idx], pos[1][idx])
        # Get restaurant name and cluster info
        restaurant = metadatas[idx].get('title', 'Unknown')
        cluster = clusters[idx]
        text = f"Restaurant: {restaurant}\nCluster: {cluster}"
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    # Save the visualization
    plt.savefig(f"roadfood_{EDITION}_embeddings_3d.png", dpi=300, bbox_inches='tight')
    print(f"Visualization saved as roadfood_{EDITION}_embeddings_3d.png")
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by the three components: {explained_variance}")
    print(f"Total explained variance: {sum(explained_variance):.2%}")
    
    # Print PCA coordinate ranges
    print("\nPCA coordinate ranges:")
    for i in range(3):
        min_val = embeddings_3d[:, i].min()
        max_val = embeddings_3d[:, i].max()
        print(f"PCA{i+1}: Min = {min_val:.4f}, Max = {max_val:.4f}, Range = {max_val - min_val:.4f}")
    
    # Normalize PCA dimensions to -1 to 1 range
    normalized_pca = np.zeros_like(embeddings_3d)
    for i in range(3):
        min_val = embeddings_3d[:, i].min()
        max_val = embeddings_3d[:, i].max()
        # First normalize to 0-1 range
        temp_normalized = (embeddings_3d[:, i] - min_val) / (max_val - min_val)
        # Then scale to -1 to 1 range
        normalized_pca[:, i] = (temp_normalized * 2) - 1
        # Round to 2 decimal places
        normalized_pca[:, i] = np.round(normalized_pca[:, i], 2)
    
    # Print cluster information
    print("\nCluster distribution:")
    unique, counts = np.unique(clusters, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} documents")
    
    # Create a DataFrame to save as CSV
    print("Creating CSV file with embedding data...")
    
    # Get RGB values from the colormap
    cmap = plt.cm.get_cmap('tab10')
    rgb_values = []
    
    for cluster_id in clusters:
        # Get RGB values (0-1 range) for each cluster
        rgba = cmap(cluster_id)
        # Convert to 0-255 range and keep only RGB (drop alpha)
        rgb = [int(255 * c) for c in rgba[:3]]
        rgb_values.append(rgb)
    
    # For Rc, Gc, Bc we need 0-1 normalized values first
    normalized_for_rgb = np.zeros_like(embeddings_3d)
    for i in range(3):
        min_val = embeddings_3d[:, i].min()
        max_val = embeddings_3d[:, i].max()
        normalized_for_rgb[:, i] = (embeddings_3d[:, i] - min_val) / (max_val - min_val)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Title': [metadata.get('title', 'Unknown') for metadata in metadatas],
        'CityState': [extract_city_state(metadata.get('address', 'Unknown')) for metadata in metadatas],
        'cluster_ID': clusters,
        'pca1': normalized_pca[:, 0],
        'pca2': normalized_pca[:, 1],
        'pca3': normalized_pca[:, 2],
        'R': [rgb[0] for rgb in rgb_values],
        'G': [rgb[1] for rgb in rgb_values],
        'B': [rgb[2] for rgb in rgb_values],
        # Add continuous RGB values based on PCA dimensions (0-255 range)
        'Rc': (normalized_for_rgb[:, 0] * 255).astype(int),
        'Gc': (normalized_for_rgb[:, 1] * 255).astype(int),
        'Bc': (normalized_for_rgb[:, 2] * 255).astype(int)
    })
    
    # Save to CSV
    csv_path = os.path.join("data", f"roadfood_{EDITION}_clusters.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to {csv_path}")

def extract_city_state(address):
    """
    Extract city and state from an address string.
    Example: "117 Pearl St. Noank, CT" -> "Noank, CT"
    """
    if address == 'Unknown':
        return 'Unknown'
    
    # Try to find the part after the last period and before the end
    # This often separates the street address from city/state
    parts = address.split('.')
    if len(parts) > 1:
        city_state_part = parts[-1].strip()
    else:
        # If no period, just use the whole address
        city_state_part = address
    
    # Look for the comma that typically separates city and state
    if ',' in city_state_part:
        # Extract from the beginning of this part to the end
        return city_state_part.strip()
    else:
        # If no comma found, return the best guess or Unknown
        return city_state_part.strip() if city_state_part.strip() else 'Unknown'

if __name__ == "__main__":
    visualize_embeddings_3d() 