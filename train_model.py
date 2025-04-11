import os
import argparse
import torch
from torch.utils.data import random_split, DataLoader
from model import RPlanDataset, ConditionalVAE, train_model, save_model
import matplotlib.pyplot as plt
import numpy as np

def train_and_evaluate(args):
    # Set device (force GPU usage if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, using CPU.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = RPlanDataset(args.csv, args.img_dir)
    
    # Split dataset
    train_size = int(len(dataset) * (1 - args.val_split))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False  # Optimizes GPU performance
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model and move to GPU
    condition_dim = len(dataset.numerical_cols) + len(dataset.binary_cols)
    model = ConditionalVAE(condition_dim, args.latent_dim).to(device)
    
    # Train model
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        lr=args.learning_rate, 
        device=device
    )
    
    # Save model
    save_model(model, os.path.join(args.output_dir, 'vae_model.pt'))
    
    # Generate sample floor plans
    model.eval()
    with torch.no_grad():
        # Create different conditions
        conditions = []
        
        # Small apartment
        small_apt = dataset.normalize_condition({
            'num_rooms': 3,
            'bedrooms': 1,
            'bathrooms': 1,
            'kitchen': 1,
            'living_room': 1,
            'total_area_sqm': 50
        })
        conditions.append(('Small Apartment', small_apt))
        
        # Medium apartment
        medium_apt = dataset.normalize_condition({
            'num_rooms': 5,
            'bedrooms': 2,
            'bathrooms': 1,
            'kitchen': 1,
            'living_room': 1,
            'dining_room': 1,
            'total_area_sqm': 80
        })
        conditions.append(('Medium Apartment', medium_apt))
        
        # Large house
        large_house = dataset.normalize_condition({
            'num_rooms': 8,
            'bedrooms': 3,
            'bathrooms': 2,
            'kitchen': 1,
            'living_room': 1,
            'dining_room': 1,
            'study_room': 1,
            'storage_room': 1,
            'total_area_sqm': 150
        })
        conditions.append(('Large House', large_house))
        
        # Generate and save samples
        fig, axes = plt.subplots(len(conditions), 3, figsize=(15, 5 * len(conditions)))
        
        for i, (name, condition) in enumerate(conditions):
            for j in range(3):
                # Generate with different random seeds
                z = torch.randn(1, args.latent_dim, device=device)
                generated = model.decoder(z, condition.to(device))[0, 0].cpu().numpy()
                
                axes[i, j].imshow(generated, cmap='gray')
                axes[i, j].set_title(f"{name} - Sample {j+1}")
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'sample_generations.png'))
        plt.close()
    
    print(f"Training complete. Model and samples saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train Floor Plan Generator')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with floor plan metadata')
    parser.add_argument('--img_dir', type=str, default=None, help='Directory with floor plan images')
    parser.add_argument('--output_dir', type=str, default='model_output', help='Output directory for model and samples')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    train_and_evaluate(args)

if __name__ == '__main__':
    main()
