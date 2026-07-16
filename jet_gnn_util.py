from torch_geometric.data import Data
import numpy as np
import torch


def create_graph_data(jet_images, labels=None, max_nodes=100, consider_all_nodes=True):
    """
    Convert jet images to graph format for GNN using PyTorch Geometric format
    Args:
        jet_images: Array of shape (N, 30, 30, 1) containing jet images
        labels: Array of labels corresponding to the images
        max_nodes: Maximum number of nodes per graph (default: 100)
    Returns:
        List of PyTorch Geometric Data objects containing node features, edge indices, and labels
    """
    data_list = []
    if labels is None:
        labels = np.zeros(len(jet_images))
    # Normalize the input images
    jet_images = (jet_images - jet_images.mean()) / (jet_images.std() + 1e-8)

    # Iterate over each image and its corresponding label
    for i, (image, label) in enumerate(zip(jet_images, labels)):
        # Get image dimensions (30x30x1)
        height, width = image.shape[:2]

        # Create node features and their 2D coordinates
        node_features = []
        node_coords = []

        # Get non-zero pixels and their coordinates
        for row in range(height):
            for col in range(width):
                intensity = image[row, col, 0]
                if intensity > 0 or consider_all_nodes:  # Only consider non-zero pixels
                    node_features.append(intensity)
                    node_coords.append((row, col))

        node_features = np.array(node_features)
        node_coords = np.array(node_coords)

        # Select top nodes by intensity if needed
        if len(node_features) > max_nodes:
            # Get indices of top max_nodes pixels by intensity
            top_indices = np.argsort(node_features)[-max_nodes:]
            node_features = node_features[top_indices]
            node_coords = node_coords[top_indices]

        n_nodes = len(node_features)

        # Create adjacency matrix based on spatial proximity
        adj_matrix = np.zeros((n_nodes, n_nodes))

        # For each node, connect to its k nearest neighbors
        for i in range(n_nodes):
            # Calculate distances to all other nodes
            distances = np.sqrt(np.sum((node_coords - node_coords[i]) ** 2, axis=1))
            # Connect to k nearest neighbors (excluding self)
            k = min(8, n_nodes - 1)  # Connect to up to 8 nearest neighbors
            nearest_indices = np.argsort(distances)[1 : k + 1]  # Skip first (self)
            adj_matrix[i, nearest_indices] = 1
            adj_matrix[nearest_indices, i] = 1  # Make it symmetric

        # Convert to PyTorch tensors and create edge_index
        x = torch.FloatTensor(node_features).view(-1, 1)  # Shape: (n_nodes, 1)
        edge_index = torch.nonzero(
            torch.FloatTensor(adj_matrix)
        ).t()  # Shape: (2, num_edges)
        y = torch.tensor(label, dtype=torch.float)
        # Validate graph structure
        if edge_index.shape[1] == 0:
            print(f"Warning: Graph {i} has no edges")
            continue

        if x.shape[0] == 0:
            print(f"Warning: Graph {i} has no nodes")
            continue

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    print(f"Created {len(data_list)} graphs")
    print(
        f"Average number of nodes: {np.mean([data.num_nodes for data in data_list]):.1f}"
    )
    print(
        f"Average number of edges: {np.mean([data.num_edges for data in data_list]):.1f}"
    )

    return data_list
