"""Cluster KAN experts by specialization patterns.

This module clusters experts based on their variable usage patterns,
predictions, or learned representations to identify specialized subgroups
that can be combined more effectively than simple averaging.

Key Features:
- Clustering by variable importance (which features each expert relies on)
- Clustering by prediction similarity (how experts group in output space)
- Clustering by weight similarity (structural similarity)
- Automatic cluster count selection

Reference:
- Plan Section: Extension 1 - Clustering by Variable Usage
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class ExpertClusterer:
    """Cluster KAN experts by specialization patterns.

    Args:
        ensemble: KANExpertEnsemble instance
        method: Clustering method ('kmeans', 'hierarchical', 'dbscan')

    Example:
        >>> clusterer = ExpertClusterer(ensemble)
        >>> labels = clusterer.cluster_by_importance(X_val, y_val, n_clusters=3)
        >>> print(f"Expert 0 belongs to cluster {labels[0]}")
    """

    def __init__(self, ensemble, method: str = 'kmeans'):
        """Initialize clusterer.

        Args:
            ensemble: KANExpertEnsemble instance
            method: Clustering algorithm
        """
        self.ensemble = ensemble
        self.method = method

        if len(ensemble.experts) == 0:
            raise ValueError("Ensemble has no trained experts")

    def cluster_by_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_clusters: Optional[int] = None,
        importance_method: str = 'permutation'
    ) -> np.ndarray:
        """Cluster experts by their variable importance patterns.

        Experts that rely on similar input features are grouped together.

        Args:
            X: Input data (N, input_dim)
            y: Target data (N, output_dim)
            n_clusters: Number of clusters (None = auto-select)
            importance_method: Importance computation method

        Returns:
            Cluster labels (n_experts,)
        """
        try:
            from .variable_importance import VariableImportanceAnalyzer
        except ImportError:
            from variable_importance import VariableImportanceAnalyzer

        analyzer = VariableImportanceAnalyzer(self.ensemble)

        # Get importance for each expert
        importance_matrix = []
        for expert_idx in range(len(self.ensemble.experts)):
            if importance_method == 'permutation':
                importance = analyzer.compute_permutation_importance(
                    X, y, n_repeats=5, expert_idx=expert_idx
                )
            elif importance_method == 'gradient':
                importance = analyzer.compute_gradient_importance(X, expert_idx=expert_idx)
            elif importance_method == 'weight':
                importance = analyzer.compute_weight_importance(expert_idx=expert_idx)
            else:
                raise ValueError(f"Unknown importance method: {importance_method}")

            importance_matrix.append(importance)

        importance_matrix = np.array(importance_matrix)  # (n_experts, input_dim)

        # Cluster
        labels = self._cluster(importance_matrix, n_clusters)

        return labels

    def cluster_by_predictions(
        self,
        X: torch.Tensor,
        n_clusters: Optional[int] = None,
        metric: str = 'correlation'
    ) -> np.ndarray:
        """Cluster experts by prediction similarity.

        Experts with similar predictions are grouped together.

        Args:
            X: Input data (N, input_dim)
            n_clusters: Number of clusters (None = auto-select)
            metric: Distance metric ('correlation', 'euclidean')

        Returns:
            Cluster labels (n_experts,)
        """
        X = X.to(self.ensemble.device)

        # Get predictions from all experts
        predictions = []
        with torch.no_grad():
            for expert in self.ensemble.experts:
                pred = expert(X).cpu().numpy()
                predictions.append(pred.flatten())

        predictions = np.array(predictions)  # (n_experts, N*output_dim)

        # Compute similarity matrix
        if metric == 'correlation':
            # Use correlation as similarity
            corr_matrix = np.corrcoef(predictions)
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(corr_matrix)
            # Fill diagonal
            np.fill_diagonal(distance_matrix, 0)

            # Use precomputed distances
            labels = self._cluster(
                distance_matrix,
                n_clusters,
                metric='precomputed' if self.method == 'hierarchical' else metric
            )
        else:
            labels = self._cluster(predictions, n_clusters, metric=metric)

        return labels

    def cluster_by_weights(
        self,
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """Cluster experts by weight vector similarity.

        Experts with similar learned weights are grouped together.

        Args:
            n_clusters: Number of clusters (None = auto-select)

        Returns:
            Cluster labels (n_experts,)
        """
        # Extract weight vectors from each expert
        weight_vectors = []
        for expert in self.ensemble.experts:
            weights = []
            for param in expert.parameters():
                weights.append(param.detach().cpu().numpy().flatten())
            weight_vectors.append(np.concatenate(weights))

        weight_vectors = np.array(weight_vectors)  # (n_experts, n_params)

        # Cluster
        labels = self._cluster(weight_vectors, n_clusters)

        return labels

    def _cluster(
        self,
        features: np.ndarray,
        n_clusters: Optional[int] = None,
        metric: str = 'euclidean'
    ) -> np.ndarray:
        """Perform clustering.

        Args:
            features: Feature matrix (n_experts, n_features) or distance matrix
            n_clusters: Number of clusters (None = auto-select)
            metric: Distance metric

        Returns:
            Cluster labels (n_experts,)
        """
        n_experts = features.shape[0]

        # Validate input data
        if np.any(np.isnan(features)):
            raise ValueError("Feature matrix contains NaN values")
        if np.any(np.isinf(features)):
            raise ValueError("Feature matrix contains Inf values")

        # Check if all features are identical (can cause clustering issues)
        if features.shape[0] > 1:
            feature_std = np.std(features, axis=0)
            if np.all(feature_std < 1e-10):
                print("Warning: All experts have nearly identical features. Assigning to single cluster.")
                return np.zeros(n_experts, dtype=int)

        # Auto-select number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._select_n_clusters(features, metric=metric)

        # Ensure n_clusters is valid
        if n_clusters >= n_experts:
            print(f"Warning: n_clusters ({n_clusters}) >= n_experts ({n_experts}). Using n_clusters={max(1, n_experts-1)}")
            n_clusters = max(1, n_experts - 1)

        if n_clusters == 1:
            return np.zeros(n_experts, dtype=int)

        # Perform clustering
        try:
            if self.method == 'kmeans':
                if metric == 'precomputed':
                    raise ValueError("KMeans does not support precomputed distances")

                # Use n_init='auto' for sklearn 1.4+ compatibility, fallback to 10 for older versions
                try:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                except TypeError:
                    # Older sklearn versions don't support n_init='auto'
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

                labels = clusterer.fit_predict(features)

            elif self.method == 'hierarchical':
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric=metric if metric != 'correlation' else 'euclidean',
                    linkage='average'
                )
                labels = clusterer.fit_predict(features)

            elif self.method == 'dbscan':
                # DBSCAN doesn't require n_clusters
                clusterer = DBSCAN(eps=0.5, min_samples=2, metric=metric)
                labels = clusterer.fit_predict(features)

            else:
                raise ValueError(f"Unknown clustering method: {self.method}")

        except Exception as e:
            print(f"Error during clustering: {e}")
            print(f"Feature matrix shape: {features.shape}")
            print(f"Feature matrix stats: min={np.min(features):.6f}, max={np.max(features):.6f}, mean={np.mean(features):.6f}")
            # Fallback: assign all to same cluster
            print("Falling back to single cluster assignment")
            return np.zeros(n_experts, dtype=int)

        return labels

    def _select_n_clusters(
        self,
        features: np.ndarray,
        max_clusters: Optional[int] = None,
        metric: str = 'euclidean'
    ) -> int:
        """Auto-select optimal number of clusters using silhouette score.

        Args:
            features: Feature matrix
            max_clusters: Maximum clusters to try
            metric: Distance metric

        Returns:
            Optimal number of clusters
        """
        n_experts = features.shape[0]

        if max_clusters is None:
            max_clusters = min(n_experts // 2, 6)

        if max_clusters < 2:
            return 1

        best_score = -1
        best_k = 2

        for k in range(2, max_clusters + 1):
            if k >= n_experts:
                break

            try:
                if self.method == 'kmeans':
                    if metric == 'precomputed':
                        continue  # Skip
                    # Use n_init='auto' for sklearn 1.4+ compatibility
                    try:
                        clusterer = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    except TypeError:
                        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(features)
                elif self.method == 'hierarchical':
                    clusterer = AgglomerativeClustering(n_clusters=k, metric=metric, linkage='average')
                    labels = clusterer.fit_predict(features)
                else:
                    continue

                # Compute silhouette score
                if len(np.unique(labels)) > 1:
                    if metric == 'precomputed':
                        score = silhouette_score(features, labels, metric='precomputed')
                    else:
                        score = silhouette_score(features, labels)

                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception as e:
                # Silently continue on errors during cluster selection
                continue

        return best_k

    def get_cluster_summary(
        self,
        labels: np.ndarray,
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None
    ) -> Dict:
        """Get summary statistics for each cluster.

        Args:
            labels: Cluster labels (n_experts,)
            X: Input data (optional, for computing cluster performance)
            y: Target data (optional)

        Returns:
            Dictionary with cluster statistics
        """
        n_clusters = len(np.unique(labels[labels >= 0]))

        summary = {
            'n_clusters': n_clusters,
            'cluster_sizes': {},
            'cluster_members': {}
        }

        for cluster_id in range(n_clusters):
            members = np.where(labels == cluster_id)[0]
            summary['cluster_sizes'][cluster_id] = len(members)
            summary['cluster_members'][cluster_id] = members.tolist()

        # Compute cluster performance if data provided
        if X is not None and y is not None:
            X = X.to(self.ensemble.device)
            y = y.to(self.ensemble.device)

            summary['cluster_performance'] = {}

            for cluster_id in range(n_clusters):
                members = summary['cluster_members'][cluster_id]

                # Average predictions within cluster
                cluster_preds = []
                with torch.no_grad():
                    for expert_idx in members:
                        pred = self.ensemble.experts[expert_idx](X)
                        cluster_preds.append(pred)

                cluster_preds = torch.stack(cluster_preds)
                cluster_mean = cluster_preds.mean(dim=0)

                # Compute MSE
                mse = torch.nn.MSELoss()(cluster_mean, y).item()
                summary['cluster_performance'][cluster_id] = mse

        return summary

    def visualize_clusters(
        self,
        labels: np.ndarray,
        features: np.ndarray,
        title: str = "Expert Clustering"
    ):
        """Visualize clustering in 2D using PCA.

        Args:
            labels: Cluster labels
            features: Feature matrix used for clustering
            title: Plot title
        """
        from sklearn.decomposition import PCA

        # Project to 2D
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            var_explained = pca.explained_variance_ratio_
        else:
            features_2d = features
            var_explained = [1.0, 0.0]

        # Plot
        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(labels[labels >= 0])

        for label in unique_labels:
            mask = labels == label
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                label=f'Cluster {label}',
                s=100,
                alpha=0.7
            )

        # Mark outliers (label = -1 for DBSCAN)
        if -1 in labels:
            mask = labels == -1
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                label='Outliers',
                s=100,
                marker='x',
                c='black'
            )

        plt.xlabel(f'PC1 ({var_explained[0]:.1%} var)')
        plt.ylabel(f'PC2 ({var_explained[1]:.1%} var)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "section1"))
    sys.path.insert(0, str(Path(__file__).parent))

    from expert_training import KANExpertEnsemble

    print("="*70)
    print("Testing Expert Clustering")
    print("="*70)

    # Generate synthetic data with different feature importances
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 200
    n_features = 5

    X = torch.randn(n_samples, n_features)
    # Different experts might learn different feature dependencies
    y = (2.0 * X[:, 0] + 1.5 * torch.sin(X[:, 2]) + 0.5 * X[:, 1]).reshape(-1, 1)

    # Train ensemble
    ensemble = KANExpertEnsemble(
        input_dim=n_features,
        hidden_dim=10,
        output_dim=1,
        depth=3,
        n_experts=8,
        kan_variant='rbf'
    )

    print("\nTraining ensemble...")
    results = ensemble.train_experts(X, y, epochs=100, lr=0.01, verbose=False)
    print(f"Training complete. Mean loss: {np.mean(results['individual_losses']):.6f}")

    # Cluster experts
    clusterer = ExpertClusterer(ensemble, method='kmeans')

    print("\n" + "="*70)
    print("Clustering Results")
    print("="*70)

    # Cluster by importance
    print("\n1. Clustering by Variable Importance:")
    labels_importance = clusterer.cluster_by_importance(
        X, y, n_clusters=3, importance_method='permutation'
    )
    summary = clusterer.get_cluster_summary(labels_importance, X, y)
    print(f"   Number of clusters: {summary['n_clusters']}")
    for cluster_id, members in summary['cluster_members'].items():
        size = summary['cluster_sizes'][cluster_id]
        perf = summary['cluster_performance'][cluster_id]
        print(f"   Cluster {cluster_id}: {size} experts, MSE = {perf:.6f}")
        print(f"     Members: {members}")

    # Cluster by predictions
    print("\n2. Clustering by Prediction Similarity:")
    labels_pred = clusterer.cluster_by_predictions(X, n_clusters=3)
    summary = clusterer.get_cluster_summary(labels_pred, X, y)
    print(f"   Number of clusters: {summary['n_clusters']}")
    for cluster_id, members in summary['cluster_members'].items():
        size = summary['cluster_sizes'][cluster_id]
        perf = summary['cluster_performance'][cluster_id]
        print(f"   Cluster {cluster_id}: {size} experts, MSE = {perf:.6f}")

    print("\nClustering analysis complete!")
