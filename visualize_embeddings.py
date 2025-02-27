import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from langchain_chroma import Chroma

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
    
    print(f"First metadata: {metadatas[0]}")
    print(f"Embeddings type: {type(embeddings)}, shape: {embeddings.shape}")
    
    # Check if embeddings are None or empty using NumPy's size attribute
    if embeddings is None or embeddings.size == 0:
        print("No embeddings found in the database.")
        return
    
    print(f"Found {len(embeddings)} embeddings.")
    print(f"First embedding shape: {embeddings[0].shape}")
    
    # No need to convert to numpy array since it already is one
    embeddings_array = embeddings
    
    # Reduce dimensionality to 3D using PCA
    print("Reducing dimensionality to 3D...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings_array)
    
    # Perform k-means clustering
    print(f"Performing k-means clustering with {NUM_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    clusters = kmeans.fit_predict(embeddings_array)
    
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
    
    # Print cluster information
    print("\nCluster distribution:")
    unique, counts = np.unique(clusters, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} documents")

if __name__ == "__main__":
    visualize_embeddings_3d() 