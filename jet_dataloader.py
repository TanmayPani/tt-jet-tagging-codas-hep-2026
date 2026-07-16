import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from jet_util import (
    load_images,
    anti_kt_clustering,
    image_to_constituents,
    canonicalize_jet,
    pixelize,
    smear_pt,
    smear_pos,
)


def prepare_constituents(
    image,
    augment=False,
    canonicalize=True,
    pt_smear=0.05,
    pos_smear=0.05,
    pt_min=0.0,
    rng=None,
):
    """Turn a jet image into a (optionally augmented and aligned) constituent cloud.

    The shared front-end for all jet datasets. Smearing (train-time augmentation)
    is applied to the raw constituents *before* canonicalization, so the alignment
    responds to the smeared physics.

    Args:
        image (np.ndarray): 2D or 3D (channel-last) jet image.
        augment (bool): Apply pT/position smearing (use only on the train split).
        canonicalize (bool): Align to the canonical orientation.
        pt_smear (float): Relative energy-resolution width (0 disables).
        pos_smear (float): Angular-resolution width in eta-phi units (0 disables).
        pt_min (float): Keep only pixels with intensity strictly above this.
        rng (np.random.Generator | None): Optional RNG for reproducible smearing.

    Returns:
        np.ndarray: (N, 3) array of [eta, phi, pt].
    """
    cons = image_to_constituents(image, pt_min=pt_min)
    if augment:
        if rng is None:
            rng = np.random.default_rng()
        if pt_smear:
            cons = smear_pt(cons, sigma=pt_smear, rng=rng)
        if pos_smear:
            cons = smear_pos(cons, sigma=pos_smear, rng=rng)
    if canonicalize:
        cons = canonicalize_jet(cons)
    return cons


def process_jet_to_clusters(image, R=0.4, pt_min=0.1, max_clusters=2):
    """Process a single jet image into anti-kt clusters.

    Args:
        image (np.ndarray): Jet image array
        R (float): Jet radius parameter
        pt_min (float): Minimum pT threshold
        max_clusters (int): Maximum number of clusters to keep

    Returns:
        np.ndarray: Array of cluster features [pt, eta, phi] for top N clusters by pT
    """
    # Get clusters from anti-kt algorithm
    clusters = anti_kt_clustering(image, R=R, pt_min=pt_min)

    # Convert clusters to feature array
    cluster_features = []
    for cluster in clusters:
        cluster = np.array(cluster)
        pt = np.sum(cluster[:, 2])
        eta = np.mean(cluster[:, 0])  # eta is first column
        phi = np.mean(cluster[:, 1])  # phi is second column
        features = np.array([pt, eta, phi])
        cluster_features.append(features)

    # Convert to numpy array
    cluster_features = np.array(cluster_features)

    if len(cluster_features) == 0:
        # If no clusters found, return array of zeros
        return np.zeros((max_clusters, 3))

    # Sort clusters by pT (first column) in descending order
    pt_order = np.argsort(-cluster_features[:, 0])
    cluster_features = cluster_features[pt_order]

    # Take top N clusters
    if len(cluster_features) > max_clusters:
        cluster_features = cluster_features[:max_clusters]
    elif len(cluster_features) < max_clusters:
        # Pad with zeros if we have fewer than max_clusters
        padding = np.zeros((max_clusters - len(cluster_features), 3))
        cluster_features = np.vstack([cluster_features, padding])

    return cluster_features


class JetClusterDataset(Dataset):
    def __init__(self, images, labels=None, R=0.4, pt_min=0.1, max_clusters=10):
        """Initialize the dataset.

        Args:
            images (np.ndarray): Array of jet images
            labels (np.ndarray): Array of labels
            R (float): Jet radius parameter
            pt_min (float): Minimum pT threshold
            max_clusters (int): Maximum number of clusters to keep per jet
        """
        self.images = images
        if labels is not None:
            self.labels = torch.as_tensor(labels, dtype=torch.float32)
        else:
            # set dummy labels
            self.labels = torch.ones(len(images))
        self.R = R
        self.pt_min = pt_min
        self.max_clusters = max_clusters

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Process image to clusters
        clusters = process_jet_to_clusters(
            self.images[idx], self.R, self.pt_min, self.max_clusters
        )

        # Convert to tensor
        clusters = torch.as_tensor(
            clusters, dtype=torch.float32
        )  # Shape: [max_clusters, 3]

        return clusters, self.labels[idx]


def get_dataloaders(batch_size=32, R=0.4, pt_min=0.1, max_clusters=10, num_workers=4):
    """Create dataloaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders
        R (float): Jet radius parameter
        pt_min (float): Minimum pT threshold
        max_clusters (int): Maximum number of clusters to keep per jet
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load all data using the existing function
    (
        X_train_images,
        y_train,
        train_ids,
        X_val_images,
        y_val,
        val_ids,
        X_test_images,
        test_ids,
    ) = load_images()

    # Create datasets
    train_dataset = JetClusterDataset(
        X_train_images, y_train, R=R, pt_min=pt_min, max_clusters=max_clusters
    )
    val_dataset = JetClusterDataset(
        X_val_images, y_val, R=R, pt_min=pt_min, max_clusters=max_clusters
    )
    test_dataset = JetClusterDataset(
        X_test_images, R=R, pt_min=pt_min, max_clusters=max_clusters
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class JetImageDataset(Dataset):
    """Canonicalized (and optionally augmented) jet images for CNNs.

    Each item is a channel-first tensor of shape [1, n, n], built by extracting
    constituents, augmenting/aligning them via :func:`prepare_constituents`, and
    re-pixelizing onto a fresh grid.
    """

    def __init__(
        self,
        images,
        labels=None,
        augment=False,
        canonicalize=True,
        pt_smear=0.05,
        pos_smear=0.05,
        pt_min=0.0,
        n=30,
        center=15,
        pixel=0.1,
        normalize="l1",
    ):
        self.images = images
        if labels is not None:
            self.labels = torch.as_tensor(labels, dtype=torch.float32)
        else:
            self.labels = torch.ones(len(images))
        self.augment = augment
        self.canonicalize = canonicalize
        self.pt_smear = pt_smear
        self.pos_smear = pos_smear
        self.pt_min = pt_min
        self.n = n
        self.center = center
        self.pixel = pixel
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cons = prepare_constituents(
            self.images[idx],
            augment=self.augment,
            canonicalize=self.canonicalize,
            pt_smear=self.pt_smear,
            pos_smear=self.pos_smear,
            pt_min=self.pt_min,
        )
        image = pixelize(cons, n=self.n, center=self.center, pixel=self.pixel)
        if self.normalize == "l1":
            total = image.sum()
            if total > 0:
                image = image / total
        image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, n, n]
        return image, self.labels[idx]


def _knn_edge_index(coords, k):
    """Build a symmetric k-nearest-neighbour edge_index from 2D coordinates."""
    n = len(coords)
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    k_eff = min(k, n - 1)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    neighbors = np.argsort(dist, axis=1)[:, :k_eff]  # [n, k_eff]
    src = np.repeat(np.arange(n), k_eff)
    dst = neighbors.reshape(-1)
    edge = np.stack([src, dst])
    edge = np.concatenate([edge, edge[::-1]], axis=1)  # make undirected
    return torch.as_tensor(edge, dtype=torch.long)


class JetGraphDataset(Dataset):
    """Canonicalized (and optionally augmented) jets as graphs for GNNs.

    Each item is a PyTorch Geometric ``Data`` object whose nodes are jet
    constituents. Node features ``x`` are [eta, phi, pt], ``pos`` holds the
    [eta, phi] coordinates, and edges connect each node to its ``k`` nearest
    neighbours in the eta-phi plane. Batch these with
    ``torch_geometric.loader.DataLoader``.
    """

    def __init__(
        self,
        images,
        labels=None,
        augment=False,
        canonicalize=True,
        pt_smear=0.05,
        pos_smear=0.05,
        pt_min=0.0,
        k=8,
        max_nodes=None,
    ):
        self.images = images
        if labels is not None:
            self.labels = torch.as_tensor(labels, dtype=torch.float32)
        else:
            self.labels = torch.ones(len(images))
        self.augment = augment
        self.canonicalize = canonicalize
        self.pt_smear = pt_smear
        self.pos_smear = pos_smear
        self.pt_min = pt_min
        self.k = k
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cons = prepare_constituents(
            self.images[idx],
            augment=self.augment,
            canonicalize=self.canonicalize,
            pt_smear=self.pt_smear,
            pos_smear=self.pos_smear,
            pt_min=self.pt_min,
        )
        # Keep the highest-pT nodes if a cap is set.
        if self.max_nodes is not None and len(cons) > self.max_nodes:
            top = np.argsort(-cons[:, 2])[: self.max_nodes]
            cons = cons[top]

        coords = cons[:, :2]
        x = torch.as_tensor(cons, dtype=torch.float32)  # [N, 3] = [eta, phi, pt]
        pos = torch.as_tensor(coords, dtype=torch.float32)  # [N, 2]
        edge_index = _knn_edge_index(coords, self.k)
        y = self.labels[idx].view(1)
        return Data(x=x, edge_index=edge_index, pos=pos, y=y)


def _build_split_loaders(
    dataset_cls, loader_cls, batch_size, num_workers, augment, dataset_kwargs
):
    """Load all splits and wrap them in train/val/test loaders.

    Only the training split is augmented; val/test are deterministic.
    """
    (
        X_train_images,
        y_train,
        _train_ids,
        X_val_images,
        y_val,
        _val_ids,
        X_test_images,
        _test_ids,
    ) = load_images()

    train_dataset = dataset_cls(
        X_train_images, y_train, augment=augment, **dataset_kwargs
    )
    val_dataset = dataset_cls(X_val_images, y_val, augment=False, **dataset_kwargs)
    test_dataset = dataset_cls(X_test_images, None, augment=False, **dataset_kwargs)

    train_loader = loader_cls(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = loader_cls(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = loader_cls(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def get_image_dataloaders(
    batch_size=32, augment=True, num_workers=4, **dataset_kwargs
):
    """Create CNN image dataloaders for train/val/test (train augmented by default)."""
    return _build_split_loaders(
        JetImageDataset, DataLoader, batch_size, num_workers, augment, dataset_kwargs
    )


def get_graph_dataloaders(
    batch_size=32, augment=True, num_workers=4, **dataset_kwargs
):
    """Create GNN graph dataloaders for train/val/test (train augmented by default)."""
    return _build_split_loaders(
        JetGraphDataset, GeoDataLoader, batch_size, num_workers, augment, dataset_kwargs
    )
