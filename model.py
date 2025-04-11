import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import io

class RPlanDataset(Dataset):
    def __init__(self, csv_path, img_dir=None, transform=None, img_size=256):
        """
        Args:
            csv_path: Path to the CSV file with floor plan metadata
            img_dir: Directory with floor plan images (if None, uses image_path from CSV)
            transform: Optional transform to be applied on images
            img_size: Size to resize images to
        """
        self.data_frame = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        
        # Define transformations
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
        # Define numerical columns for normalization
        self.numerical_cols = [
            'num_rooms', 'bedrooms', 'bathrooms', 'corridors', 
            'total_area_sqm', 'wall_thickness_cm', 'num_doors', 
            'num_windows', 'floor_level', 'avg_room_distance_m'
        ]
        
        # Define binary columns
        self.binary_cols = [
            'kitchen', 'living_room', 'dining_room', 
            'study_room', 'balcony', 'storage_room'
        ]
        
        # Initialize scalers for numerical features
        self.scalers = {}
        for col in self.numerical_cols:
            scaler = MinMaxScaler()
            self.data_frame[col] = scaler.fit_transform(self.data_frame[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path
        img_path = self.data_frame.iloc[idx]['image_path']
        if self.img_dir is not None:
            img_path = os.path.join(self.img_dir, os.path.basename(img_path))
        
        # Load image
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            image = torch.zeros((1, self.img_size, self.img_size))
        
        # Get condition features
        condition = []
        
        # Add numerical features
        for col in self.numerical_cols:
            condition.append(float(self.data_frame.iloc[idx][col]))
        
        # Add binary features
        for col in self.binary_cols:
            condition.append(float(self.data_frame.iloc[idx][col]))
        
        # Convert to tensor
        condition = torch.tensor(condition, dtype=torch.float32)
        
        return {'image': image, 'condition': condition}
    
    def normalize_condition(self, condition_dict):
        """Normalize a condition dictionary for model input"""
        normalized = []
        
        # Normalize numerical features
        for col in self.numerical_cols:
            if col in condition_dict:
                # Use the fitted scaler to transform the value
                value = self.scalers[col].transform([[condition_dict[col]]])[0][0]
            else:
                # Use a default value if not provided
                value = 0.5  # Middle of the normalized range
            normalized.append(value)
        
        # Add binary features
        for col in self.binary_cols:
            if col in condition_dict:
                value = float(condition_dict[col])
            else:
                value = 0.0  # Default to not present
            normalized.append(value)
        
        return torch.tensor([normalized], dtype=torch.float32)

class Encoder(nn.Module):
    def __init__(self, condition_dim, latent_dim=128):
        super(Encoder, self).__init__()
        
        # Image processing layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Condition processing
        self.condition_net = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # Calculate the size of the flattened features
        # For 256x256 input, after 4 layers of stride 2, we get 16x16 feature maps
        flattened_size = 256 * 16 * 16
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(flattened_size + 512, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size + 512, latent_dim)
    
    def forward(self, x, condition):
        # Process image
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Process condition
        condition = self.condition_net(condition)
        
        # Concatenate image features and condition
        x_combined = torch.cat([x, condition], dim=1)
        
        # Get mean and log variance
        mu = self.fc_mu(x_combined)
        logvar = self.fc_logvar(x_combined)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class Decoder(nn.Module):
    def __init__(self, condition_dim, latent_dim=128):
        super(Decoder, self).__init__()
        
        # Process latent vector and condition
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 16 * 16),
            nn.ReLU()
        )
        
        # Transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
    
    def forward(self, z, condition):
        # Concatenate latent vector with condition
        z_condition = torch.cat([z, condition], dim=1)
        
        # Process combined latent vector and condition
        x = self.fc(z_condition)
        
        # Reshape to 3D tensor for transposed convolutions
        x = x.view(x.size(0), 256, 16, 16)
        
        # Upsampling with transposed convolutions
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        
        return x

class ConditionalVAE(nn.Module):
    def __init__(self, condition_dim, latent_dim=128):
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(condition_dim, latent_dim)
        self.decoder = Decoder(condition_dim, latent_dim)
        self.latent_dim = latent_dim
    
    def forward(self, x, condition):
        # Encode
        mu, logvar = self.encoder(x, condition)
        
        # Reparameterize
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decoder(z, condition)
        
        return recon_x, mu, logvar
    
    def generate(self, condition, num_samples=1):
        """Generate floor plans based on condition"""
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, self.latent_dim)
            
            # Decode with condition
            generated = self.decoder(z, condition)
            
            return generated

def train_model(model, train_loader, val_loader=None, epochs=50, lr=0.0001, device='cpu'):
    """Train the CVAE model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Move model to device
    model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Get data
            images = batch['image'].to(device)
            conditions = batch['condition'].to(device)
            
            # Forward pass
            recon_images, mu, logvar = model(images, conditions)
            
            # Calculate loss
            recon_loss = F.binary_cross_entropy(recon_images, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    conditions = batch['condition'].to(device)
                    
                    recon_images, mu, logvar = model(images, conditions)
                    
                    recon_loss = F.binary_cross_entropy(recon_images, images, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            print(f"Validation Loss: {avg_val_loss:.4f}")
    
    return model

def save_model(model, path):
    """Save the trained model"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path, condition_dim=16, latent_dim=128):
    """Load a trained model"""
    model = ConditionalVAE(condition_dim, latent_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def generate_floorplan(model, condition_dict, dataset):
    """Generate a floor plan based on condition dictionary"""
    # Normalize condition
    condition = dataset.normalize_condition(condition_dict)
    
    # Generate floor plan
    with torch.no_grad():
        generated = model.generate(condition)[0, 0].numpy()
    
    # Calculate space efficiency metrics
    efficiency = calculate_efficiency_metrics(generated, condition_dict)
    
    return generated, efficiency

def calculate_efficiency_metrics(floor_plan, condition_dict):
    """Calculate space efficiency metrics for the generated floor plan"""
    # This is a simplified version - in a real implementation,
    # you would analyze the floor plan to calculate these metrics
    
    # For now, we'll return some dummy metrics
    total_area = condition_dict.get('total_area_sqm', 100)
    num_rooms = condition_dict.get('num_rooms', 4)
    
    # Calculate wall area (approximation)
    wall_pixels = (floor_plan < 0.3).sum()
    total_pixels = floor_plan.size
    wall_ratio = wall_pixels / total_pixels
    
    # Calculate metrics
    metrics = {
        'Space Efficiency': 0.85 - (wall_ratio * 0.5),  # Higher is better
        'Room Ratio': min(1.0, total_area / (num_rooms * 20)),  # Area per room
        'Flow Score': 0.75 + (condition_dict.get('corridors', 1) * 0.05)  # Movement flow
    }
    
    return metrics

def prepare_dataset(csv_path, img_dir=None, batch_size=32, val_split=0.2):
    """Prepare dataset and dataloaders"""
    # Create dataset
    dataset = RPlanDataset(csv_path, img_dir)
    
    # Split into train and validation
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return dataset, train_loader, val_loader

def main():
    """Main function to train the model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataset
    csv_path = "floorplan_dataset_improved.csv"  # Update with your CSV path
    img_dir = "floorplan_dataset"  # Update with your image directory
    
    dataset, train_loader, val_loader = prepare_dataset(
        csv_path, img_dir, batch_size=32, val_split=0.2
    )
    
    # Create model
    condition_dim = len(dataset.numerical_cols) + len(dataset.binary_cols)
    model = ConditionalVAE(condition_dim, latent_dim=128)
    
    # Train model
    model = train_model(
        model, train_loader, val_loader, 
        epochs=50, lr=0.0001, device=device
    )
    
    # Save model
    save_model(model, "vae_model.pt")

if __name__ == "__main__":
    main()


