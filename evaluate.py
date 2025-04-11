import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import RPlanDataset, ConditionalVAE, load_model, calculate_space_efficiency

def evaluate_model(model, test_loader, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get a batch of test data
    batch = next(iter(test_loader))
    images = batch['image'].to(device)
    conditions = batch['condition'].to(device)
    original_params = batch['original_params']
    
    # Reconstruct images
    with torch.no_grad():
        recon_images, _, _ = model(images, conditions)
    
    # Generate new images
    with torch.no_grad():
        generated = model.generate(conditions)
    
    # Convert to numpy for visualization
    original = images.cpu().numpy()
    reconstructed = recon_images.cpu().numpy()
    generated = generated.cpu().numpy()
    
    # Plot results
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Original
        axes[i, 0].imshow(original[i, 0], cmap='gray')
        axes[i, 0].set_title(f"Original - {original_params['num_rooms'][i]} rooms, {original_params['total_area_sqm'][i]} sqm")
        axes[i, 0].axis('off')
        
        # Reconstructed
        axes[i, 1].imshow(reconstructed[i, 0], cmap='gray')
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis('off')
        
        # Generated
        axes[i, 2].imshow(generated[i, 0], cmap='gray')
        axes[i, 2].set_title("Generated")
        axes[i, 2].axis('off')
        
        # Calculate efficiency metrics for generated floor plan
        efficiency = calculate_space_efficiency(
            generated[i, 0], 
            {k: original_params[k][i] for k in original_params}
        )
        
        # Add efficiency metrics as text
        efficiency_text = "\n".join([f"{k}: {v:.2f}" for k, v in efficiency.items()])
        axes[i, 2].text(1.05, 0.5, efficiency_text, transform=axes[i, 2].transAxes)
    
    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    plt.show()
    
    print("Evaluation results saved to evaluation_results.png")

def main(args):
    # Create dataset
    dataset = RPlanDataset(args.csv_path)
    
    # Create data loader
    test_loader = DataLoader(dataset, batch_size=args.num_samples)
    
    # Load model
    model = load_model(args.model_path)
    
    # Evaluate model
    evaluate_model(model, test_loader, num_samples=args.num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAE model for floor plan generation")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with floor plan data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    main(args)