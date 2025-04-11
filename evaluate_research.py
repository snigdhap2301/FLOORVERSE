import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import json
from datetime import datetime

# Import model components - assuming these are available in your model.py file
from model import RPlanDataset, ConditionalVAE, load_model, calculate_efficiency_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAE model for floor plan generation research paper")
    
    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with floor plan data")
    parser.add_argument("--img_dir", type=str, default=None, help="Directory with floor plan images")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for evaluation")
    
    # Model comparison (optional)
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    parser.add_argument("--model_paths", type=str, nargs='+', default=[], help="Paths to models for comparison")
    parser.add_argument("--model_names", type=str, nargs='+', default=[], help="Names for models in comparison")
    
    # Figure customization
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument("--paper_mode", action="store_true", help="Generate publication-quality figures")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def create_output_dir(output_dir):
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def compute_metrics(model, dataloader, device):
    """Compute reconstruction loss and KL divergence for the dataset"""
    model.eval()
    total_recon_loss = 0
    total_kl_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing metrics"):
            images = batch['image'].to(device)
            conditions = batch['condition'].to(device)
            
            # Forward pass
            recon_images, mu, logvar = model(images, conditions)
            
            # Calculate losses
            recon_loss = F.binary_cross_entropy(recon_images, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_samples += images.size(0)
    
    # Average losses
    avg_recon_loss = total_recon_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples
    total_loss = avg_recon_loss + avg_kl_loss
    
    return {
        'reconstruction_loss': avg_recon_loss,
        'kl_divergence': avg_kl_loss,
        'total_loss': total_loss
    }

def visualize_reconstructions(model, dataloader, device, output_dir, num_samples=5, dpi=300, paper_mode=False):
    """Visualize original and reconstructed floor plans"""
    model.eval()
    
    # Get a batch of data
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    conditions = batch['condition'].to(device)
    
    # Reconstruct images
    with torch.no_grad():
        recon_images, _, _ = model(images, conditions)
    
    # Convert to numpy for visualization
    original = images.cpu().numpy()
    reconstructed = recon_images.cpu().numpy()
    
    # Create custom colormap for floor plans
    colors = [(1, 1, 1), (0.8, 0.8, 0.8), (0.5, 0.5, 0.5), (0, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("floor_plan_cmap", colors, N=256)
    
    # Plot results
    if paper_mode:
        # Publication-quality figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2 * num_samples))
        plt.subplots_adjust(hspace=0.3)
        
        for i in range(num_samples):
            # Original
            axes[i, 0].imshow(original[i, 0], cmap=cmap)
            axes[i, 0].set_title("Original", fontsize=10)
            axes[i, 0].axis('off')
            
            # Reconstructed
            axes[i, 1].imshow(reconstructed[i, 0], cmap=cmap)
            axes[i, 1].set_title("Reconstructed", fontsize=10)
            axes[i, 1].axis('off')
        
        fig.suptitle("Original vs. Reconstructed Floor Plans", fontsize=12, y=0.98)
    else:
        # Standard figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
        
        for i in range(num_samples):
            # Original
            axes[i, 0].imshow(original[i, 0], cmap=cmap)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')
            
            # Reconstructed
            axes[i, 1].imshow(reconstructed[i, 0], cmap=cmap)
            axes[i, 1].set_title("Reconstructed")
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstructions.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "reconstructions.pdf"), bbox_inches='tight')
    plt.close(fig)
    
    return original, reconstructed

def visualize_latent_space(model, dataloader, device, output_dir, dpi=300, paper_mode=False):
    """Visualize the latent space using t-SNE and PCA"""
    model.eval()
    
    # Collect latent vectors and conditions
    latent_vectors = []
    condition_values = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting latent vectors"):
            images = batch['image'].to(device)
            conditions = batch['condition'].to(device)
            
            # Get latent vectors
            mu, _ = model.encoder(images, conditions)
            latent_vectors.append(mu.cpu().numpy())
            
            # Store condition values for coloring
            condition_values.append(conditions.cpu().numpy())
    
    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    condition_values = np.concatenate(condition_values, axis=0)
    
    # Use PCA to reduce dimensionality to 50 before t-SNE (if latent space is large)
    if latent_vectors.shape[1] > 50:
        pca = PCA(n_components=50)
        latent_vectors_reduced = pca.fit_transform(latent_vectors)
    else:
        latent_vectors_reduced = latent_vectors
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_vectors_reduced)
    
    # Apply PCA
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vectors)
    
    # Create visualizations
    if paper_mode:
        # Publication-quality figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # t-SNE plot
        scatter1 = axes[0].scatter(
            latent_tsne[:, 0], 
            latent_tsne[:, 1], 
            c=condition_values[:, 0],  # Color by first condition (num_rooms)
            cmap='viridis', 
            alpha=0.7,
            s=20  # Smaller point size for publication
        )
        axes[0].set_title('t-SNE Visualization', fontsize=10)
        axes[0].set_xlabel('t-SNE Dimension 1', fontsize=8)
        axes[0].set_ylabel('t-SNE Dimension 2', fontsize=8)
        axes[0].tick_params(axis='both', which='major', labelsize=8)
        cbar1 = fig.colorbar(scatter1, ax=axes[0], shrink=0.8)
        cbar1.ax.tick_params(labelsize=8)
        cbar1.set_label('Number of Rooms', fontsize=8)
        
        # PCA plot
        scatter2 = axes[1].scatter(
            latent_pca[:, 0], 
            latent_pca[:, 1], 
            c=condition_values[:, 0],  # Color by first condition (num_rooms)
            cmap='viridis', 
            alpha=0.7,
            s=20  # Smaller point size for publication
        )
        axes[1].set_title('PCA Visualization', fontsize=10)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=8)
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=8)
        axes[1].tick_params(axis='both', which='major', labelsize=8)
        cbar2 = fig.colorbar(scatter2, ax=axes[1], shrink=0.8)
        cbar2.ax.tick_params(labelsize=8)
        cbar2.set_label('Number of Rooms', fontsize=8)
        
        fig.suptitle("Latent Space Visualization", fontsize=12, y=0.98)
    else:
        # Standard figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # t-SNE plot
        scatter1 = axes[0].scatter(
            latent_tsne[:, 0], 
            latent_tsne[:, 1], 
            c=condition_values[:, 0],  # Color by first condition (num_rooms)
            cmap='viridis', 
            alpha=0.7
        )
        axes[0].set_title('t-SNE Visualization of Latent Space')
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        fig.colorbar(scatter1, ax=axes[0], label='Number of Rooms')
        
        # PCA plot
        scatter2 = axes[1].scatter(
            latent_pca[:, 0], 
            latent_pca[:, 1], 
            c=condition_values[:, 0],  # Color by first condition (num_rooms)
            cmap='viridis', 
            alpha=0.7
        )
        axes[1].set_title('PCA Visualization of Latent Space')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        fig.colorbar(scatter2, ax=axes[1], label='Number of Rooms')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_space.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "latent_space.pdf"), bbox_inches='tight')
    plt.close(fig)
    
    return latent_vectors, latent_tsne, latent_pca

def generate_conditional_samples(model, dataset, device, output_dir, num_variations=5, dpi=300, paper_mode=False):
    """Generate floor plans with varying conditions"""
    model.eval()
    
    # Define different conditions to test
    conditions = [
        {
            'name': 'Small Apartment',
            'params': {
                'num_rooms': 3,
                'bedrooms': 1,
                'bathrooms': 1,
                'corridors': 1,
                'kitchen': 1,
                'living_room': 1,
                'dining_room': 0,
                'study_room': 0,
                'balcony': 0,
                'storage_room': 0,
                'total_area_sqm': 50,
                'wall_thickness_cm': 15,
                'num_doors': 4,
                'num_windows': 3,
                'floor_level': 1,
                'avg_room_distance_m': 2
            }
        },
        {
            'name': 'Medium Apartment',
            'params': {
                'num_rooms': 5,
                'bedrooms': 2,
                'bathrooms': 1,
                'corridors': 1,
                'kitchen': 1,
                'living_room': 1,
                'dining_room': 1,
                'study_room': 0,
                'balcony': 0,
                'storage_room': 0,
                'total_area_sqm': 80,
                'wall_thickness_cm': 15,
                'num_doors': 6,
                'num_windows': 5,
                'floor_level': 1,
                'avg_room_distance_m': 3
            }
        },
        {
            'name': 'Large House',
            'params': {
                'num_rooms': 8,
                'bedrooms': 3,
                'bathrooms': 2,
                'corridors': 2,
                'kitchen': 1,
                'living_room': 1,
                'dining_room': 1,
                'study_room': 1,
                'balcony': 0,
                'storage_room': 1,
                'total_area_sqm': 150,
                'wall_thickness_cm': 20,
                'num_doors': 10,
                'num_windows': 8,
                'floor_level': 1,
                'avg_room_distance_m': 4
            }
        }
    ]
    
    # Create custom colormap for floor plans
    colors = [(1, 1, 1), (0.8, 0.8, 0.8), (0.5, 0.5, 0.5), (0, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("floor_plan_cmap", colors, N=256)
    
    # Generate samples for each condition
    all_efficiency_metrics = {}
    
    for condition in conditions:
        name = condition['name']
        params = condition['params']
        
        # Normalize condition
        normalized_condition = dataset.normalize_condition(params)
        normalized_condition = normalized_condition.to(device)
        
        # Generate multiple samples with the same condition
        samples = []
        metrics = []
        
        with torch.no_grad():
            for i in range(num_variations):
                # Sample from latent space
                z = torch.randn(1, model.latent_dim, device=device)
                
                # Generate floor plan
                generated = model.decoder(z, normalized_condition)
                generated_np = generated[0, 0].cpu().numpy()
                
                samples.append(generated_np)
                
                # Calculate efficiency metrics
                efficiency = calculate_efficiency_metrics(generated_np, params)
                metrics.append(efficiency)
        
        # Plot generated samples
        if paper_mode:
            # Publication-quality figure
            fig, axes = plt.subplots(1, num_variations, figsize=(2 * num_variations, 2))
            
            for i, sample in enumerate(samples):
                axes[i].imshow(sample, cmap=cmap)
                axes[i].set_title(f"V{i+1}", fontsize=8)
                axes[i].axis('off')
            
            fig.suptitle(f"{name}", fontsize=10)
        else:
            # Standard figure
            fig, axes = plt.subplots(1, num_variations, figsize=(4 * num_variations, 4))
            
            for i, sample in enumerate(samples):
                axes[i].imshow(sample, cmap=cmap)
                axes[i].set_title(f"Variation {i+1}")
                axes[i].axis('off')
            
            plt.suptitle(f"{name} - Floor Plan Variations", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_variations.png"), dpi=dpi, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_variations.pdf"), bbox_inches='tight')
        plt.close(fig)
        
        # Store metrics
        all_efficiency_metrics[name] = metrics
    
    # Plot efficiency metrics comparison
    plot_efficiency_metrics(all_efficiency_metrics, output_dir, dpi, paper_mode)
    
    return all_efficiency_metrics

def plot_efficiency_metrics(all_metrics, output_dir, dpi=300, paper_mode=False):
    """Plot efficiency metrics for different conditions"""
    # Prepare data for plotting
    conditions = list(all_metrics.keys())
    metrics_names = list(all_metrics[conditions[0]][0].keys())
    
    # Calculate average metrics for each condition
    avg_metrics = {}
    for condition in conditions:
        avg_metrics[condition] = {}
        for metric in metrics_names:
            values = [m[metric] for m in all_metrics[condition]]
            avg_metrics[condition][metric] = np.mean(values)
    
    # Create bar plot for each metric
    for metric in metrics_names:
        if paper_mode:
            # Publication-quality figure
            plt.figure(figsize=(5, 3))
            
            values = [avg_metrics[condition][metric] for condition in conditions]
            bars = plt.bar(conditions, values, color=plt.cm.viridis(np.linspace(0, 0.8, len(conditions))))
            
            plt.title(f"{metric}", fontsize=10)
            plt.ylabel("Score", fontsize=8)
            plt.ylim(0, max(values) * 1.2)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            
            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.2f}",
                    ha='center',
                    fontsize=7,
                    rotation=0
                )
        else:
            # Standard figure
            plt.figure(figsize=(10, 6))
            
            values = [avg_metrics[condition][metric] for condition in conditions]
            bars = plt.bar(conditions, values, color=plt.cm.viridis(np.linspace(0, 0.8, len(conditions))))
            
            plt.title(f"Average {metric} by Floor Plan Type", fontsize=14)
            plt.ylabel(f"{metric} Score", fontsize=12)
            plt.ylim(0, max(values) * 1.2)
            
            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.2f}",
                    ha='center',
                    fontsize=10,
                    rotation=0
                )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metric_{metric.lower().replace(' ', '_')}.png"), dpi=dpi, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"metric_{metric.lower().replace(' ', '_')}.pdf"), bbox_inches='tight')
        plt.close()
    
    # Create radar chart for comparing all metrics across conditions
    if paper_mode:
        # Publication-quality figure
        fig = plt.figure(figsize=(5, 4))
    else:
        # Standard figure
        fig = plt.figure(figsize=(10, 8))
        
    ax = fig.add_subplot(111, polar=True)
    
    # Number of metrics
    N = len(metrics_names)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Plot for each condition
    for i, condition in enumerate(conditions):
        values = [avg_metrics[condition][metric] for metric in metrics_names]
        values += values[:1]  # Close the loop
        
        # Normalize values to 0-1 range for radar chart
        max_values = [max([avg_metrics[c][m] for c in conditions]) for m in metrics_names]
        normalized_values = [v / max_v if max_v > 0 else 0 for v, max_v in zip(values, max_values + max_values[:1])]
        
        ax.plot(angles, normalized_values, linewidth=2, label=condition, color=plt.cm.viridis(i / len(conditions) * 0.8))
        ax.fill(angles, normalized_values, alpha=0.1, color=plt.cm.viridis(i / len(conditions) * 0.8))
    
    # Set labels
    ax.set_xticks(angles[:-1])
    if paper_mode:
        ax.set_xticklabels(metrics_names, fontsize=8)
    else:
        ax.set_xticklabels(metrics_names)
    
    # Remove radial labels
    ax.set_yticklabels([])
    
    # Add legend
    if paper_mode:
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)
        plt.title("Efficiency Metrics", fontsize=10, y=1.1)
    else:
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Efficiency Metrics Comparison", fontsize=15, y=1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_radar_chart.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "metrics_radar_chart.pdf"), bbox_inches='tight')
    plt.close()

def analyze_latent_space_traversal(model, dataset, device, output_dir, dpi=300, paper_mode=False):
    """Analyze latent space by traversing along principal components"""
    model.eval()
    
    # Define a base condition
    base_condition = {
        'num_rooms': 5,
        'bedrooms': 2,
        'bathrooms': 1,
        'corridors': 1,
        'kitchen': 1,
        'living_room': 1,
        'dining_room': 1,
        'study_room': 0,
        'balcony': 0,
        'storage_room': 0,
        'total_area_sqm': 80,
        'wall_thickness_cm': 15,
        'num_doors': 6,
        'num_windows': 5,
        'floor_level': 1,
        'avg_room_distance_m': 3
    }
    
    # Normalize condition
    normalized_condition = dataset.normalize_condition(base_condition).to(device)
    
    # Create a random base latent vector
    torch.manual_seed(42)  # For reproducibility
    base_z = torch.randn(1, model.latent_dim, device=device)
    
    # Define traversal range
    traversal_range = np.linspace(-3, 3, 7)
    
    # Create custom colormap for floor plans
    colors = [(1, 1, 1), (0.8, 0.8, 0.8), (0.5, 0.5, 0.5), (0, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("floor_plan_cmap", colors, N=256)
    
    # Traverse along first 5 dimensions
    num_dims = min(5, model.latent_dim)
    
    if paper_mode:
        # Publication-quality figure
        fig, axes = plt.subplots(num_dims, len(traversal_range), figsize=(len(traversal_range), num_dims))
    else:
        # Standard figure
        fig, axes = plt.subplots(num_dims, len(traversal_range), figsize=(len(traversal_range) * 2, num_dims * 2))
    
    with torch.no_grad():
        for dim in range(num_dims):
            for i, val in enumerate(traversal_range):
                # Create a copy of the base latent vector
                z = base_z.clone()
                
                # Modify the dimension
                z[0, dim] = val
                
                # Generate floor plan
                generated = model.decoder(z, normalized_condition)
                generated_np = generated[0, 0].cpu().numpy()
                
                # Plot
                axes[dim, i].imshow(generated_np, cmap=cmap)
                if paper_mode:
                    axes[dim, i].set_title(f"{val:.1f}", fontsize=7)
                else:
                    axes[dim, i].set_title(f"Dim {dim+1}: {val:.1f}")
                axes[dim, i].axis('off')
    
    if paper_mode:
        # Add dimension labels on the left
        for dim in range(num_dims):
            axes[dim, 0].text(-0.5, 0.5, f"z{dim+1}", fontsize=8, ha='center', va='center', transform=axes[dim, 0].transAxes)
        
        fig.suptitle("Latent Space Traversal", fontsize=10, y=0.98)
    else:
        plt.suptitle("Latent Space Traversal", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_traversal.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "latent_traversal.pdf"), bbox_inches='tight')
    plt.close(fig)

def create_metrics_table(metrics, output_dir, dpi=300, paper_mode=False):
    """Create a publication-ready table of metrics"""
    # Create a DataFrame for the metrics
    df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    # Format metric names
    df['Metric'] = df['Metric'].apply(lambda x: x.replace('_', ' ').title())
    
    # Create a styled table
    if paper_mode:
        # Publication-quality figure
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        # Standard figure
        fig, ax = plt.subplots(figsize=(8, 4))
        
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.6, 0.3]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    if paper_mode:
        table.set_fontsize(8)
        table.scale(1.2, 1.2)
    else:
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
    
    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:  # Data rows
            if j == 0:  # Metric names
                cell.set_text_props(ha='left', weight='bold')
            else:  # Values
                cell.set_text_props(ha='right')
                # Format numbers
                if isinstance(cell.get_text().get_text(), str) and cell.get_text().get_text().replace('.', '', 1).isdigit():
                    cell.get_text().set_text(f"{float(cell.get_text().get_text()):.4f}")
            
            # Alternate row colors
            if i % 2 == 1:
                cell.set_facecolor('#D9E1F2')
            else:
                cell.set_facecolor('#E9EDF4')
    
    if paper_mode:
        plt.title("Model Performance Metrics", fontsize=10, pad=10)
    else:
        plt.title("Model Performance Metrics", fontsize=16, pad=20)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_table.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "metrics_table.pdf"), bbox_inches='tight')
    plt.close()

def create_research_figure(output_dir, dpi=300):
    """Create a comprehensive research figure combining multiple visualizations"""
    # Create figure with grid layout
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # A. Reconstructions (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    recon_img = plt.imread(os.path.join(output_dir, "reconstructions.png"))
    ax1.imshow(recon_img)
    ax1.set_title("A. Original vs. Reconstructed", fontsize=12)
    ax1.axis('off')
    
    # B. Latent Space (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    latent_img = plt.imread(os.path.join(output_dir, "latent_space.png"))
    ax2.imshow(latent_img)
    ax2.set_title("B. Latent Space Visualization", fontsize=12)
    ax2.axis('off')
    
    # C. Latent Traversal (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    traversal_img = plt.imread(os.path.join(output_dir, "latent_traversal.png"))
    ax3.imshow(traversal_img)
    ax3.set_title("C. Latent Space Traversal", fontsize=12)
    ax3.axis('off')
    
    # D. Small Apartment Variations (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    small_apt_img = plt.imread(os.path.join(output_dir, "small_apartment_variations.png"))
    ax4.imshow(small_apt_img)
    ax4.set_title("D. Small Apartment Variations", fontsize=12)
    ax4.axis('off')
    
    # E. Medium Apartment Variations (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])
    medium_apt_img = plt.imread(os.path.join(output_dir, "medium_apartment_variations.png"))
    ax5.imshow(medium_apt_img)
    ax5.set_title("E. Medium Apartment Variations", fontsize=12)
    ax5.axis('off')
    
    # F. Large House Variations (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    large_house_img = plt.imread(os.path.join(output_dir, "large_house_variations.png"))
    ax6.imshow(large_house_img)
    ax6.set_title("F. Large House Variations", fontsize=12)
    ax6.axis('off')
    
    # G. Efficiency Metrics (bottom left and middle)
    ax7 = fig.add_subplot(gs[2, 0:2])
    metrics_img = plt.imread(os.path.join(output_dir, "metrics_radar_chart.png"))
    ax7.imshow(metrics_img)
    ax7.set_title("G. Efficiency Metrics Comparison", fontsize=12)
    ax7.axis('off')
    
    # H. Space Efficiency Metric (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    try:
        space_eff_img = plt.imread(os.path.join(output_dir, "metric_space_efficiency.png"))
        ax8.imshow(space_eff_img)
        ax8.set_title("H. Space Efficiency by Floor Plan Type", fontsize=12)
    except FileNotFoundError:
        # Fallback to another metric if this one doesn't exist
        flow_score_img = plt.imread(os.path.join(output_dir, "metric_flow_score.png"))
        ax8.imshow(flow_score_img)
        ax8.set_title("H. Flow Score by Floor Plan Type", fontsize=12)
    ax8.axis('off')
    
    # Add main title
    fig.suptitle("Floor Plan Generator: Model Evaluation and Analysis", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "research_figure.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "research_figure.pdf"), bbox_inches='tight')
    plt.close()

def create_summary_report(metrics, output_dir):
    """Create a summary report of all evaluation metrics"""
    # Create a summary dictionary
    summary = {
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_metrics': metrics
    }
    
    # Save as JSON
    with open(os.path.join(output_dir, "evaluation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Create a markdown report
    with open(os.path.join(output_dir, "evaluation_report.md"), 'w') as f:
        f.write("# Floor Plan Generator Evaluation Report\n\n")
        f.write(f"**Date:** {summary['evaluation_date']}\n\n")
        
        f.write("## Model Performance Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, value in metrics.items():
            f.write(f"| {key.replace('_', ' ').title()} | {value:.4f} |\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("### Reconstructions\n\n")
        f.write("![Reconstructions](reconstructions.png)\n\n")
        
        f.write("### Latent Space\n\n")
        f.write("![Latent Space](latent_space.png)\n\n")
        
        f.write("### Latent Space Traversal\n\n")
        f.write("![Latent Traversal](latent_traversal.png)\n\n")
        
        f.write("### Efficiency Metrics\n\n")
        f.write("![Efficiency Metrics](metrics_radar_chart.png)\n\n")
        
        f.write("### Generated Samples\n\n")
        f.write("#### Small Apartment\n\n")
        f.write("![Small Apartment](small_apartment_variations.png)\n\n")
        
        f.write("#### Medium Apartment\n\n")
        f.write("![Medium Apartment](medium_apartment_variations.png)\n\n")
        
        f.write("#### Large House\n\n")
        f.write("![Large House](large_house_variations.png)\n\n")
        
        f.write("## Research Figure\n\n")
        f.write("![Research Figure](research_figure.png)\n\n")

def compare_models(model_paths, model_names, csv_path, img_dir, output_dir, batch_size=32, device="cuda", dpi=300, paper_mode=False):
    """Compare multiple models and visualize their performance"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = RPlanDataset(csv_path, img_dir)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # Evaluate each model
    all_metrics = {}
    
    for model_name, model_path in zip(model_names, model_paths):
        print(f"Evaluating model: {model_name}")
        
        # Load model
        model = load_model(model_path)
        model = model.to(device)
        
        # Compute metrics
        metrics = compute_metrics(model, dataloader, device)
        all_metrics[model_name] = metrics
    
    # Create comparison table
    create_comparison_table(all_metrics, output_dir, dpi, paper_mode)
    
    # Create comparison charts
    create_comparison_charts(all_metrics, output_dir, dpi, paper_mode)
    
    return all_metrics

def create_comparison_table(all_metrics, output_dir, dpi=300, paper_mode=False):
    """Create a comparison table of metrics across models"""
    # Get all unique metrics
    all_metric_keys = set()
    for model_metrics in all_metrics.values():
        all_metric_keys.update(model_metrics.keys())
    
    # Create DataFrame
    data = []
    for metric in sorted(all_metric_keys):
        row = {'Metric': metric.replace('_', ' ').title()}
        for model_name, model_metrics in all_metrics.items():
            row[model_name] = model_metrics.get(metric, np.nan)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv(os.path.join(output_dir, "metrics_comparison.csv"), index=False)
    
    # Create a styled table visualization
    if paper_mode:
        # Publication-quality figure
        fig, ax = plt.subplots(figsize=(6, len(data) * 0.4 + 1))
    else:
        # Standard figure
        fig, ax = plt.subplots(figsize=(10, len(data) * 0.5 + 1))
        
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    if paper_mode:
        table.set_fontsize(8)
        table.scale(1.2, 1.2)
    else:
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
    
    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:  # Data rows
            if j == 0:  # Metric names
                cell.set_text_props(ha='left', weight='bold')
            else:  # Values
                cell.set_text_props(ha='right')
                # Format numbers
                if isinstance(cell.get_text().get_text(), str) and cell.get_text().get_text().replace('.', '', 1).isdigit():
                    cell.get_text().set_text(f"{float(cell.get_text().get_text()):.4f}")
            
            # Alternate row colors
            if i % 2 == 1:
                cell.set_facecolor('#D9E1F2')
            else:
                cell.set_facecolor('#E9EDF4')
    
    if paper_mode:
        plt.title("Model Performance Comparison", fontsize=10, pad=10)
    else:
        plt.title("Model Performance Comparison", fontsize=16, pad=20)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison_table.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "metrics_comparison_table.pdf"), bbox_inches='tight')
    plt.close()
    
    return df

def create_comparison_charts(all_metrics, output_dir, dpi=300, paper_mode=False):
    """Create bar charts comparing metrics across models"""
    # Prepare data for plotting
    metrics = set()
    for model_metrics in all_metrics.values():
        metrics.update(model_metrics.keys())
    
    # Create bar chart for each metric
    for metric in sorted(metrics):
        # Get values for each model
        models = []
        values = []
        
        for model_name, model_metrics in all_metrics.items():
            if metric in model_metrics:
                models.append(model_name)
                values.append(model_metrics[metric])
        
        if len(models) <= 1:
            continue  # Skip metrics with only one model
        
        if paper_mode:
            # Publication-quality figure
            plt.figure(figsize=(5, 3))
        else:
            # Standard figure
            plt.figure(figsize=(10, 6))
        
        # Create bar chart
        bars = plt.bar(models, values, color=plt.cm.viridis(np.linspace(0, 0.8, len(models))))
        
        if paper_mode:
            plt.title(f"{metric.replace('_', ' ').title()}", fontsize=10)
            plt.ylabel("Value", fontsize=8)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
        else:
            plt.title(f"Comparison of {metric.replace('_', ' ').title()} Across Models", fontsize=14)
            plt.ylabel(f"{metric.replace('_', ' ').title()} Value", fontsize=12)
            plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.4f}",
                ha='center',
                fontsize=7 if paper_mode else 10,
                rotation=0
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{metric.lower().replace(' ', '_')}.png"), dpi=dpi, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"comparison_{metric.lower().replace(' ', '_')}.pdf"), bbox_inches='tight')
        plt.close()
    
    # Create a radar chart comparing all models
    # Get common metrics across all models
    common_metrics = []
    for metric in metrics:
        if all(metric in model_metrics for model_metrics in all_metrics.values()):
            common_metrics.append(metric)
    
    if len(common_metrics) >= 3:  # Need at least 3 metrics for a meaningful radar chart
        # Create radar chart
        if paper_mode:
            # Publication-quality figure
            fig = plt.figure(figsize=(5, 4))
        else:
            # Standard figure
            fig = plt.figure(figsize=(10, 8))
            
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics
        N = len(common_metrics)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Get model names
        models = list(all_metrics.keys())
        
        # Plot for each model
        for i, model in enumerate(models):
            values = [all_metrics[model][metric] for metric in common_metrics]
            values += values[:1]  # Close the loop
            
            # Normalize values to 0-1 range for radar chart
            max_values = [max([all_metrics[m][metric] for m in models]) for metric in common_metrics]
            max_values += max_values[:1]  # Close the loop
            normalized_values = [v / max_v if max_v > 0 else 0 for v, max_v in zip(values, max_values)]
            
            ax.plot(angles, normalized_values, linewidth=2, label=model, color=plt.cm.viridis(i / len(models) * 0.8))
            ax.fill(angles, normalized_values, alpha=0.1, color=plt.cm.viridis(i / len(models) * 0.8))
        
        # Set labels
        ax.set_xticks(angles[:-1])
        if paper_mode:
            ax.set_xticklabels([m.replace('_', ' ').title() for m in common_metrics], fontsize=8)
        else:
            ax.set_xticklabels([m.replace('_', ' ').title() for m in common_metrics])
        
        # Remove radial labels
        ax.set_yticklabels([])
        
        # Add legend
        if paper_mode:
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)
            plt.title("Model Comparison", fontsize=10, y=1.1)
        else:
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title("Model Performance Comparison", fontsize=15, y=1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "models_radar_chart.png"), dpi=dpi, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, "models_radar_chart.pdf"), bbox_inches='tight')
        plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Results will be saved to: {output_dir}")
    
    if args.compare:
        # Compare multiple models
        if not args.model_paths or not args.model_names or len(args.model_paths) != len(args.model_names):
            print("Error: For model comparison, you must provide equal number of model_paths and model_names")
            return
        
        print(f"Comparing {len(args.model_paths)} models...")
        compare_models(
            args.model_paths, 
            args.model_names, 
            args.csv_path, 
            args.img_dir, 
            output_dir, 
            args.batch_size, 
            args.device,
            args.dpi,
            args.paper_mode
        )
    else:
        # Evaluate a single model
        # Load dataset
        dataset = RPlanDataset(args.csv_path, args.img_dir)
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        # Load model
        device = torch.device(args.device)
        model = load_model(args.model_path)
        model = model.to(device)
        print(f"Model loaded from {args.model_path} to {device}")
        
        # Compute metrics
        print("Computing evaluation metrics...")
        metrics = compute_metrics(model, dataloader, device)
        print("Metrics:", metrics)
        
        # Create metrics table
        print("Creating metrics table...")
        create_metrics_table(metrics, output_dir, args.dpi, args.paper_mode)
        
        # Visualize reconstructions
        print("Visualizing reconstructions...")
        visualize_reconstructions(model, dataloader, device, output_dir, args.num_samples, args.dpi, args.paper_mode)
        
        # Visualize latent space
        print("Visualizing latent space...")
        visualize_latent_space(model, dataloader, device, output_dir, args.dpi, args.paper_mode)
        
        # Generate conditional samples
        print("Generating conditional samples...")
        generate_conditional_samples(model, dataset, device, output_dir, 5, args.dpi, args.paper_mode)
        
        # Analyze latent space traversal
        print("Analyzing latent space traversal...")
        analyze_latent_space_traversal(model, dataset, device, output_dir, args.dpi, args.paper_mode)
        
        # Create research figure
        print("Creating research figure...")
        create_research_figure(output_dir, args.dpi)
        
        # Create summary report
        print("Creating summary report...")
        create_summary_report(metrics, output_dir)
        
        print(f"Evaluation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()

