# compare_models.py
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# Import models
from models.unetmodel import UNet
from models.transunet.transunet import TransUNet
#from models.yolov8_seg import YOLOv8Seg

# Import utilities
from dataset import get_data_loaders
from metrics import (
    iou_score, 
    dice_score, 
    boundary_f1_score, 
    ModelTimer, 
    MemoryTracker,
    count_parameters
)


# Import configuration
from config import cfg

def train_model(model_name, model, train_loader, test_loader, device, num_epochs=10, lr=0.001):
    """
    Train a model and evaluate its performance.
    
    Args:
        model_name: Name of the model
        model: Model instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        device: Device to train on
        num_epochs: Number of epochs to train
        lr: Learning rate
        
    Returns:
        dict: Dictionary containing training metrics
    """
    print(f"\nTraining {model_name}...")
    
    # Move model to device
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    accumulation_steps = cfg.accumulation_steps  # Gradient accumulation steps
    
    # Create save directories
    os.makedirs(f'./results/{model_name}', exist_ok=True)
    os.makedirs(f'./results/{model_name}/predictions', exist_ok=True)
    
    # Initialize timer and memory tracker
    timer = ModelTimer()
    memory_tracker = MemoryTracker()
    
    # Initialize metrics
    metrics = {
        'train_losses': [],
        'test_losses': [],
        'test_iou': [],
        'test_dice': [],
        'test_boundary_f1': [],
        'epoch_times': [],
        'inference_times': [],
        'memory_usage': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        memory_tracker.update()
        
        # Training phase
        model.train()
        train_loss = 0.0

        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")

        for batch_idx, (inputs, masks) in enumerate(progress_bar):
            inputs = inputs.to(device)
            masks = masks.to(device)
                        
            timer.start()
            outputs = model(inputs)

            loss = criterion(outputs, masks) / accumulation_steps
            loss.backward()

            # Track the un-scaled loss for reporting
            actual_loss = loss.detach().item() * accumulation_steps
            train_loss += actual_loss * inputs.size(0)
            
            # Update on accumulation steps or at the end of an epoch
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            timer.stop(batch_size=inputs.size(0))
            
            # Show the un-scaled loss in progress bar
            progress_bar.set_postfix(loss=actual_loss)

        scheduler.step()

        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        metrics['train_losses'].append(train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_iou = 0.0
        test_dice = 0.0
        test_boundary_f1 = 0.0
        inference_timer = ModelTimer()

        with torch.no_grad():
            for i, (inputs, masks) in enumerate(tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing")):
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                inference_timer.start()
                outputs = model(inputs)
                inference_timer.stop(batch_size=inputs.size(0))
                
                loss = criterion(outputs, masks)
                test_loss += loss.detach().item() * inputs.size(0)
                
                # Calculate metrics
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                batch_iou = iou_score(preds, masks)
                batch_dice = dice_score(preds, masks)
                
                if i == 0:
                    # Calculate on at most 2 images to save resources
                    batch_boundary_f1 = 0
                    for j in range(min(2, preds.size(0))):
                        batch_boundary_f1 += boundary_f1_score(preds[j, 0].cpu(), masks[j, 0].cpu())
                    batch_boundary_f1 /= min(2, preds.size(0))
                    test_boundary_f1 += batch_boundary_f1
                    
                    # Save predictions only on first and last epoch
                    if epoch == 0 or epoch == num_epochs - 1:
                        save_predictions(inputs, masks, preds, epoch, model_name)
                        
                test_iou += batch_iou * inputs.size(0)
                test_dice += batch_dice * inputs.size(0)
                memory_tracker.update()

        # Calculate average metrics
        test_loss = test_loss / len(test_loader.dataset)
        test_iou = test_iou / len(test_loader.dataset)
        test_dice = test_dice / len(test_loader.dataset)
        test_boundary_f1 = test_boundary_f1 / 1  # Only calculated for first batch
        
        metrics['test_losses'].append(test_loss)
        metrics['test_iou'].append(test_iou)
        metrics['test_dice'].append(test_dice)
        metrics['test_boundary_f1'].append(test_boundary_f1)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        metrics['epoch_times'].append(epoch_time)
        metrics['inference_times'].append(inference_timer.get_average_time())
        metrics['memory_usage'].append(memory_tracker.get_max_memory())
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"IoU: {test_iou:.4f}, Dice: {test_dice:.4f}, "
              f"Time: {epoch_time:.2f}s, Inference: {inference_timer.get_average_time()*1000:.2f}ms/sample")
    
    # Save trained model
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), f'saved_models/{model_name.lower()}_model.pth')
    
    # Save metrics
    save_metrics(metrics, model_name)
    
    return model, metrics

def save_predictions(images, masks, predictions, epoch, model_name):
    """Save prediction visualizations."""
    # Convert tensors to numpy arrays
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Save the first 4 images or fewer if batch size is smaller
    num_images = min(4, images.shape[0])
    
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Denormalize image
        img = images[i].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        mask = masks[i, 0]
        pred = predictions[i, 0]
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred, cmap='gray')
        axes[i, 1].set_title('Predicted Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(mask, cmap='gray')
        axes[i, 2].set_title('Ground Truth Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'./results/{model_name}/predictions/epoch_{epoch+1}.png')
    plt.close()

def save_metrics(metrics, model_name):
    """Save metrics to CSV and create plots."""
    # Save metrics to CSV
    df = pd.DataFrame({
        'Epoch': range(1, len(metrics['train_losses']) + 1),
        'Train Loss': metrics['train_losses'],
        'Test Loss': metrics['test_losses'],
        'IoU': metrics['test_iou'],
        'Dice': metrics['test_dice'],
        'Boundary F1': metrics['test_boundary_f1'],
        'Epoch Time (s)': metrics['epoch_times'],
        'Inference Time (ms/sample)': [t * 1000 for t in metrics['inference_times']],
        'Memory Usage (MB)': metrics['memory_usage']
    })
    
    df.to_csv(f'./results/{model_name}/metrics.csv', index=False)
    
    # Create accuracy metrics plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_losses'], 'b-', label='Train')
    plt.plot(metrics['test_losses'], 'r-', label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(metrics['test_iou'], 'g-', label='IoU')
    plt.plot(metrics['test_dice'], 'c-', label='Dice')
    plt.title('Segmentation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(metrics['test_boundary_f1'], 'm-', label='Boundary F1')
    plt.title('Boundary Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'./results/{model_name}/accuracy_metrics.png')
    plt.close()
    
    # Create efficiency metrics plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(metrics['epoch_times'], 'b-')
    plt.title('Epoch Training Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    
    plt.subplot(1, 3, 2)
    plt.plot([t * 1000 for t in metrics['inference_times']], 'r-')
    plt.title('Inference Time per Sample')
    plt.xlabel('Epoch')
    plt.ylabel('Time (ms)')
    
    plt.subplot(1, 3, 3)
    plt.plot(metrics['memory_usage'], 'g-')
    plt.title('Memory Usage')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    
    plt.tight_layout()
    plt.savefig(f'./results/{model_name}/efficiency_metrics.png')
    plt.close()

def compare_models(train_img_dir, train_mask_dir, test_img_dir, test_mask_dir, 
                  batch_size=16, img_size=256, num_epochs=10):
    """
    Compare multiple segmentation models on the same dataset.
    
    Args:
        train_img_dir: Directory containing training images
        train_mask_dir: Directory containing training masks
        test_img_dir: Directory containing test images
        test_mask_dir: Directory containing test masks
        batch_size: Batch size
        img_size: Input image size
        num_epochs: Number of epochs to train
        
    Returns:
        DataFrame: Comparison of model performance
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        train_img_dir, train_mask_dir, test_img_dir, test_mask_dir,
        batch_size=batch_size, img_size=img_size
    )
    
    # Initialize models
    models = {
        'UNet': UNet(n_channels=cfg.unet.n_channels, n_classes=cfg.unet.n_classes, bilinear=cfg.unet.bilinear),
        'TransUNet': TransUNet(
            img_dim=cfg.transunet.img_dim,
            in_channels=cfg.transunet.in_channels,
            out_channels=cfg.transunet.out_channels,
            head_num=cfg.transunet.head_num,
            mlp_dim=cfg.transunet.mlp_dim,
            block_num=cfg.transunet.block_num,
            patch_dim=cfg.transunet.patch_dim,
            class_num=cfg.transunet.class_num
        )
    }
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train and evaluate each model
    all_metrics = {}
    for model_name, model in models.items():
        _, metrics = train_model(
            model_name, model, train_loader, test_loader, 
            device, num_epochs=num_epochs, lr=cfg.learning_rate
        )
        all_metrics[model_name] = metrics
    
    # Create model comparison 
    compare_performance(all_metrics)
    
    return all_metrics

def compare_performance(all_metrics):
    """Create comparison visualizations across all models."""
    # Get model names
    model_names = list(all_metrics.keys())
    
    # Create results directory
    os.makedirs('results/comparison', exist_ok=True)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name in model_names:
        metrics = all_metrics[model_name]
        comparison_data.append({
            'Model': model_name,
            'Final IoU': metrics['test_iou'][-1],
            'Final Dice': metrics['test_dice'][-1],
            'Final Boundary F1': metrics['test_boundary_f1'][-1],
            'Average Epoch Time (s)': np.mean(metrics['epoch_times']),
            'Average Inference Time (ms)': np.mean(metrics['inference_times']) * 1000,
            'Peak Memory Usage (MB)': max(metrics['memory_usage'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    comparison_df.to_csv('results/comparison/model_comparison.csv', index=False)
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Create accuracy comparison plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for model_name in model_names:
        plt.plot(all_metrics[model_name]['train_losses'], label=model_name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for model_name in model_names:
        plt.plot(all_metrics[model_name]['test_iou'], label=model_name)
    plt.title('IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for model_name in model_names:
        plt.plot(all_metrics[model_name]['test_dice'], label=model_name)
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for model_name in model_names:
        plt.plot(all_metrics[model_name]['test_boundary_f1'], label=model_name)
    plt.title('Boundary F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/comparison/accuracy_comparison.png')
    plt.close()
    
    # Create efficiency comparison plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for model_name in model_names:
        plt.plot(all_metrics[model_name]['epoch_times'], label=model_name)
    plt.title('Epoch Training Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    for model_name in model_names:
        plt.plot([t * 1000 for t in all_metrics[model_name]['inference_times']], label=model_name)
    plt.title('Inference Time per Sample')
    plt.xlabel('Epoch')
    plt.ylabel('Time (ms)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    for model_name in model_names:
        plt.plot(all_metrics[model_name]['memory_usage'], label=model_name)
    plt.title('Memory Usage')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/comparison/efficiency_comparison.png')
    plt.close()
    
    # Create bar chart comparison of final metrics
    plt.figure(figsize=(15, 10))
    
    # Accuracy metrics
    plt.subplot(2, 1, 1)
    x = np.arange(len(model_names))
    width = 0.25
    
    plt.bar(x - width, [data['Final IoU'] for data in comparison_data], width, label='IoU')
    plt.bar(x, [data['Final Dice'] for data in comparison_data], width, label='Dice')
    plt.bar(x + width, [data['Final Boundary F1'] for data in comparison_data], width, label='Boundary F1')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Final Accuracy Metrics')
    plt.xticks(x, model_names)
    plt.legend()
    
    # Efficiency metrics
    plt.subplot(2, 1, 2)
    
    # Normalize values for better visualization
    inference_times = [data['Average Inference Time (ms)'] for data in comparison_data]
    max_inference = max(inference_times)
    norm_inference = [t / max_inference for t in inference_times]
    
    memory_usage = [data['Peak Memory Usage (MB)'] for data in comparison_data]
    max_memory = max(memory_usage)
    norm_memory = [m / max_memory for m in memory_usage]
    
    epoch_times = [data['Average Epoch Time (s)'] for data in comparison_data]
    max_epoch = max(epoch_times)
    norm_epoch = [t / max_epoch for t in epoch_times]
    
    plt.bar(x - width, norm_inference, width, label='Inference Time (lower is better)')
    plt.bar(x, norm_memory, width, label='Memory Usage (lower is better)')
    plt.bar(x + width, norm_epoch, width, label='Epoch Time (lower is better)')
    
    plt.xlabel('Model')
    plt.ylabel('Normalized Value (lower is better)')
    plt.title('Efficiency Metrics (Normalized)')
    plt.xticks(x, model_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/comparison/final_metrics_comparison.png')
    plt.close()
    
    return comparison_df

if __name__ == "__main__":
    #use cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define data paths
    train_img_dir = './data/ISIC-2017_Training_Data'
    train_mask_dir = './data/ISIC-2017_Training_Part1_GroundTruth'
    test_img_dir = './data/ISIC-2017_Test_v2_Data'
    test_mask_dir = './data/ISIC-2017_Test_v2_Part1_GroundTruth'

    # Run model comparison
    all_metrics = compare_models(
        train_img_dir, train_mask_dir, test_img_dir, test_mask_dir,
        batch_size=cfg.batch_size, img_size=cfg.img_size, num_epochs=cfg.num_epochs
    )