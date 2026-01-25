"""Mahalanobis OOD detector.

Why: central gate that must stay in lockstep with feature dimensions emitted by
CNN/FFN/ART. If feature shapes or params change, this must be regenerated to
avoid runtime mismatches across all entrypoints.
"""

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
            params = load(f, weights_only=False)
        
        self.class_means = params['class_means']
        # Support both full and diagonal precision matrices
        if 'precision_diag' in params:
            self.precision_diag = params['precision_diag']
            self.use_diagonal = True
        else:
            self.precision = params.get('precision')
            self.use_diagonal = False
        
        self.feature_dim = params['feature_dim']
        self.model_type = params.get('model_type', 'unknown')  # Track which model created these params
        self.num_classes = len(self.class_means)
        
        # Load class-conditional thresholds if available
        # Priority: 90th percentile (stricter) > 95th percentile > global threshold
        self.class_thresholds_90 = params.get('class_thresholds_90', {})
        self.class_thresholds_95 = params.get('class_thresholds_95', {})
        self.class_thresholds_99 = params.get('class_thresholds_99', {})
        self.class_mean_distances = params.get('class_mean_distances', {})
        self.class_std_distances = params.get('class_std_distances', {})
        
        # Keep global threshold for backward compatibility
        self.threshold_95 = params.get('threshold_95', 10.0)
        self.threshold_99 = params.get('threshold_99', 15.0)
        
        cov_type = "diagonal" if self.use_diagonal else "full"
        model_info = f" [{self.model_type.upper()} model]" if self.model_type != 'unknown' else ""
        if self.class_thresholds_90:
            print(f"✓ OOD detector loaded{model_info}: {self.num_classes} class prototypes ({cov_type} covariance)")
            print(f"  Using class-conditional thresholds (90th percentile - stricter):")
            for i in range(min(10, self.num_classes)):
                if i in self.class_thresholds_90:
                    print(f"    Class {i}: {self.class_thresholds_90[i]:.2f}")
        elif self.class_thresholds_95:
            print(f"✓ OOD detector loaded{model_info}: {self.num_classes} class prototypes ({cov_type} covariance)")
            print(f"  Using class-conditional thresholds (95th percentile):")
            for i in range(min(10, self.num_classes)):
                if i in self.class_thresholds_95:
                    print(f"    Class {i}: {self.class_thresholds_95[i]:.2f}")
        else:
            print(f"✓ OOD detector loaded{model_info}: {self.num_classes} class prototypes ({cov_type} covariance)")
            print(f"  Global threshold (95%): {self.threshold_95:.2f}")
    
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
            threshold: Mahalanobis distance threshold (uses class-conditional if None)
        
        Returns:
            belongs: True if sample is in-distribution, False if OOD
            distance: Mahalanobis distance to predicted class prototype
            min_distance: Minimum distance to any class prototype
            nearest_class: Class with minimum distance
            all_distances: Dict of distances to all class prototypes
        """
        # Select threshold: Custom > Class-90th > Class-95th > Global-95th
        if threshold is None:
            if self.class_thresholds_90 and predicted_class in self.class_thresholds_90:
                threshold = self.class_thresholds_90[predicted_class]
            elif self.class_thresholds_95 and predicted_class in self.class_thresholds_95:
                threshold = self.class_thresholds_95[predicted_class]
            else:
                threshold = self.threshold_95
        
        features = features.cpu()
        
        # Compute distance to predicted class
        distance = self.mahalanobis_distance(features, predicted_class)
        
        # Also compute distance to all classes
        all_distances = {i: self.mahalanobis_distance(features, i) 
                        for i in range(self.num_classes)}
        
        min_distance = min(all_distances.values())
        nearest_class = min(all_distances, key=all_distances.get)
        
        # Sample is in-distribution if minimum distance to any class prototype is below threshold
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
