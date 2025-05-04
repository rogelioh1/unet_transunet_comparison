# metrics.py
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
import time
import psutil

def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU (Intersection over Union) score."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()

def dice_score(pred, target, smooth=1e-6):
    """Calculate Dice score coefficient."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

def boundary_f1_score(pred, target, tolerance=3):
    """Calculate Boundary F1 score with improved boundary detection."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    
    # Get boundaries using morphological operations
    from scipy import ndimage
    
    # Dilate and erode to get boundaries
    struct_el = ndimage.generate_binary_structure(2, 2)
    pred_boundary = np.logical_xor(pred, ndimage.binary_erosion(pred, structure=struct_el))
    target_boundary = np.logical_xor(target, ndimage.binary_erosion(target, structure=struct_el))
    
    # Create distance transforms
    pred_distances = ndimage.distance_transform_edt(~pred_boundary)
    target_distances = ndimage.distance_transform_edt(~target_boundary)
    
    # Calculate precision and recall
    pred_within_target = (pred_distances <= tolerance) & target_boundary
    target_within_pred = (target_distances <= tolerance) & pred_boundary
    
    # Count true positives, false positives, and false negatives
    tp = np.sum(target_within_pred)
    fp = np.sum(pred_boundary) - tp
    fn = np.sum(target_boundary) - np.sum(pred_within_target)
    
    # Calculate precision and recall
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    
    return f1

class ModelTimer:
    """Class to measure training and inference time."""
    def __init__(self):
        self.start_time = None
        self.total_time = 0
        self.total_samples = 0
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self, batch_size=1):
        """Stop the timer and update statistics."""
        if self.start_time is None:
            return 0
            
        time_elapsed = time.time() - self.start_time
        
        self.total_time += time_elapsed
        self.total_samples += batch_size
        
        return time_elapsed
    
    def get_average_time(self):
        """Get average processing time per sample."""
        if self.total_samples == 0:
            return 0
        return self.total_time / self.total_samples

class MemoryTracker:
    """Class to track memory usage during model training/inference."""
    def __init__(self):
        self.max_memory = 0
        
    def update(self):
        """Update maximum memory usage."""
        memory_info = psutil.Process().memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        
        self.max_memory = max(self.max_memory, memory_mb)
        
        return memory_mb
    
    def get_max_memory(self):
        """Get maximum memory usage."""
        return self.max_memory

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)