import os
import pandas as pd
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import RPlanDataset
import matplotlib.pyplot as plt

def prepare_dataset(csv_path, output_dir, img_dir=None):
    """Prepare and analyze the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset safely
    try:
        dataset = RPlanDataset(csv_path, img_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if not hasattr(dataset, 'data_frame'):
        print("Error: RPlanDataset does not have a data_frame attribute.")
        return
    
    # Check dataset size
    print(f"Dataset size: {len(dataset)}")
    
    # Analyze numerical features
    numerical_stats = {}
    for col in dataset.numerical_cols:
        values = dataset.data_frame[col].values
        numerical_stats[col] = {
            'min': values.min(),
            'max': values.max(),
            'mean': values.mean(),
            'std': values.std()
        }
    
    # Analyze binary features
    binary_stats = {}
    for col in dataset.binary_cols:
        values = dataset.data_frame[col].values
        binary_stats[col] = {
            'count_1': values.sum(),
            'percentage_1': values.sum() / len(values) * 100
        }
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'dataset_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("Dataset Statistics\n=================\n\n")
        f.write(f"Total samples: {len(dataset)}\n\n")
        
        f.write("Numerical Features:\n")
        for col, stats in numerical_stats.items():
            f.write(f"  {col}:\n")
            for stat_name, stat_value in stats.items():
                f.write(f"    {stat_name}: {stat_value}\n")
            f.write("\n")
        
        f.write("Binary Features:\n")
        for col, stats in binary_stats.items():
            f.write(f"  {col}:\n")
            for stat_name, stat_value in stats.items():
                f.write(f"    {stat_name}: {stat_value}\n")
            f.write("\n")
    
    print(f"Dataset analysis saved to {stats_path}")

    # Visualize some samples
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(range(0, len(dataset), len(dataset)//6)[:6]):
        sample = dataset[idx]
        image = sample['image'][0].numpy() if 'image' in sample else torch.zeros((256, 256))
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Sample {idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_floorplans.png'))
    plt.close()
    
    print(f"Dataset samples saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare Floor Plan Dataset')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with floor plan metadata')
    parser.add_argument('--img_dir', type=str, default=None, help='Directory with floor plan images')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis', help='Output directory for analysis')
    
    args = parser.parse_args()
    prepare_dataset(args.csv, args.output_dir, args.img_dir)

if __name__ == '__main__':
    main()
