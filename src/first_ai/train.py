"""Generic training utilities for PyTorch models.

Provides reusable training loops with early stopping, mixed precision,
and comprehensive logging for classifier training.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def fit_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    loss_fn: Callable,
    device: str = 'cuda',
    scaler: Optional[torch.amp.GradScaler] = None
) -> float:
    """Train model for one epoch with optional mixed precision.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        optimizer: Optimizer instance
        loss_fn: Loss function
        device: Device to train on ('cuda' or 'cpu')
        scaler: Optional GradScaler for mixed precision training
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        X, y = batch
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # Forward pass with automatic mixed precision
            with torch.amp.autocast(device):
                yhat = model(X)
                loss = loss_fn(yhat, y)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without mixed precision
            yhat = model(X)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
    
    return train_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: Callable,
    device: str = 'cuda'
) -> tuple[float, float]:
    """Evaluate model on test set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to evaluate on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (test_loss, accuracy_percentage)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            X, y = batch
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Use autocast for inference too if available
            with torch.amp.autocast(device):
                yhat = model(X)
                loss = loss_fn(yhat, y)
            
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(yhat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    
    return test_loss, accuracy


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer,
    loss_fn: Callable,
    num_epochs: int = 10,
    patience: int = 3,
    save_path: Optional[Union[str, Path]] = None,
    device: str = 'cuda',
    use_amp: bool = True
) -> Dict[str, Any]:
    """Train model with early stopping based on test loss.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        test_loader: Test data loader  
        optimizer: Optimizer instance
        loss_fn: Loss function
        num_epochs: Maximum number of epochs
        patience: Number of epochs to wait for improvement before stopping
        save_path: Path to save best model (if None, doesn't save)
        device: Device to train on ('cuda' or 'cpu')
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Dictionary with training history: {
            'train_losses': list,
            'test_losses': list,
            'test_accuracies': list,
            'best_test_loss': float,
            'best_epoch': int,
            'stopped_epoch': int
        }
    """
    scaler = torch.amp.GradScaler(device) if use_amp else None
    
    best_test_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_losses': [],
        'test_losses': [],
        'test_accuracies': [],
        'best_test_loss': float('inf'),
        'best_epoch': 0,
        'stopped_epoch': 0
    }
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss = fit_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        
        # Evaluation phase
        test_loss, accuracy = evaluate(model, test_loader, loss_fn, device)
        
        # Record history
        history['train_losses'].append(train_loss)
        history['test_losses'].append(test_loss)
        history['test_accuracies'].append(accuracy)
        
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}, Test Accuracy = {accuracy:.2f}%")
        
        # Save model only if test loss improved
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            patience_counter = 0
            
            if save_path is not None:
                with open(save_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
                logger.info(f"  ✓ New best model saved (test loss: {test_loss:.6f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\nEarly stopping at epoch {epoch}. Best test loss: {best_test_loss:.6f}")
            history['stopped_epoch'] = epoch
            break
    else:
        history['stopped_epoch'] = num_epochs - 1
    
    history['best_test_loss'] = best_test_loss
    history['best_epoch'] = best_epoch
    
    return history
