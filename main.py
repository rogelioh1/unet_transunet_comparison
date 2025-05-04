# main.py
import os
import torch
import argparse
from config import cfg
from compare_models import compare_models
import torch.multiprocessing as mp

def main():    
    parser = argparse.ArgumentParser(description='Compare UNet, TransUNet and YOLOv8 for medical image segmentation')
    parser.add_argument('--train_dir', type=str, default='./data/ISIC-2017_Training_Data', 
                        help='Path to training images')
    parser.add_argument('--train_mask_dir', type=str, default='./data/ISIC-2017_Training_Part1_GroundTruth', 
                        help='Path to training masks')
    parser.add_argument('--test_dir', type=str, default='./data/ISIC-2017_Test_v2_Data', 
                        help='Path to test images')
    parser.add_argument('--test_mask_dir', type=str, default='./data/ISIC-2017_Test_v2_Part1_GroundTruth', 
                        help='Path to test masks')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, 
                        help='Batch size for training and testing')
    parser.add_argument('--img_size', type=int, default=cfg.img_size, 
                        help='Input image size')
    parser.add_argument('--epochs', type=int, default=cfg.num_epochs, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=cfg.learning_rate, 
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    cfg.batch_size = args.batch_size
    cfg.img_size = args.img_size
    cfg.num_epochs = args.epochs
    cfg.learning_rate = args.lr
    
    # Run model comparison
    compare_models(
        args.train_dir, args.train_mask_dir, args.test_dir, args.test_mask_dir,
        batch_size=cfg.batch_size, img_size=cfg.img_size, num_epochs=cfg.num_epochs
    )

if __name__ == '__main__':
    main()