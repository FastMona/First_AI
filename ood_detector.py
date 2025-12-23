# Out-of-Distribution detector using Mahalanobis distance
# Provides "belongs / doesn't belong" signal for MNIST digit classification

import torch
import numpy as np
from torch import load

class MahalanobisOODDetector:
    """
    Detects out-of-distribution samples using Mahalanobis distance
    to class prototypes computed from training data.
    """
    
    def __init__(self, ood_params_path='ood_params.pth'):
        """Load pre-computed OOD parameters"""
        with open(ood_params_path, 'rb') as f:
            params = load(f)
        
        self.class_means = params['class_means']
        # Support both full and diagonal precision matrices
        if 'precision_diag' in params:
            self.precision_diag = params['precision_diag']
            self.use_diagonal = True
        else:
            self.precision = params.get('precision')
            self.use_diagonal = False
        
        self.feature_dim = params['feature_dim']
        self.num_classes = len(self.class_means)
        
        # Load calibrated threshold if available
        self.threshold_95 = params.get('threshold_95', 10.0)
        self.threshold_99 = params.get('threshold_99', 15.0)
        self.mean_distance = params.get('mean_distance', 5.0)
        self.std_distance = params.get('std_distance', 2.0)
        
        cov_type = "diagonal" if self.use_diagonal else "full"
        print(f"✓ OOD detector loaded: {self.num_classes} class prototypes ({cov_type} covariance)")
        print(f"  Calibrated threshold (95%): {self.threshold_95:.2f}")
        print(f"  Mean training distance: {self.mean_distance:.2f} ± {self.std_distance:.2f}")
    
    def mahalanobis_distance(self, features, class_idx):
        """
        Compute Mahalanobis distance from features to class prototype.
        
        For diagonal covariance: distance = sqrt(sum((x - μ)^2 / σ^2))
        For full covariance: distance = sqrt((x - μ)^T Σ^(-1) (x - μ))
        """
        mean = self.class_means[class_idx]
        diff = features - mean
        
        if self.use_diagonal:
            # Diagonal Mahalanobis distance (much faster)
            distance = torch.sqrt(torch.sum(diff**2 * self.precision_diag))
        else:
            # Full Mahalanobis distance
            left = torch.mm(diff.unsqueeze(0), self.precision)
            distance = torch.mm(left, diff.unsqueeze(1)).squeeze()
            distance = torch.sqrt(distance)
        
        return distance.item()
    
    def detect(self, features, predicted_class, threshold=None):
        """
        Determine if sample belongs to the predicted class or is OOD.
        
        Args:
            features: Feature vector from model (before final layer)
            predicted_class: Predicted class from classifier
            threshold: Mahalanobis distance threshold (uses calibrated if None)
        
        Returns:
            belongs: True if sample belongs to predicted class, False if OOD
            distance: Mahalanobis distance to predicted class prototype
            min_distance: Minimum distance to any class prototype
            nearest_class: Class with minimum distance
        """
        # Use calibrated threshold if not provided
        if threshold is None:
            threshold = self.threshold_95
        
        features = features.cpu()
        
        # Compute distance to predicted class
        distance = self.mahalanobis_distance(features, predicted_class)
        
        # Also compute distance to all classes
        all_distances = {i: self.mahalanobis_distance(features, i) 
                        for i in range(self.num_classes)}
        
        min_distance = min(all_distances.values())
        nearest_class = min(all_distances, key=all_distances.get)
        
        # Sample belongs if distance to any class is below threshold
        belongs = min_distance < threshold
        
        return belongs, distance, min_distance, nearest_class, all_distances
    
    def calibrate_threshold(self, model, dataloader, percentile=95):
        """
        Calibrate threshold using validation data.
        Returns threshold at given percentile of in-distribution distances.
        """
        model.eval()
        all_distances = []
        
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch
                X = X.to('cuda')
                features = model.get_features(X)
                
                for i in range(len(y)):
                    feat = features[i].cpu()
                    label = y[i].item()
                    dist = self.mahalanobis_distance(feat, label)
                    all_distances.append(dist)
        
        threshold = np.percentile(all_distances, percentile)
        print(f"Calibrated threshold at {percentile}th percentile: {threshold:.2f}")
        print(f"  Mean distance: {np.mean(all_distances):.2f}")
        print(f"  Std distance: {np.std(all_distances):.2f}")
        
        return threshold
