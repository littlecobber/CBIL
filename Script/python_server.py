#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python Server Module
Used for communication with Unity ml-agents, processing image data and performing model inference
"""

import threading
import socket
import numpy as np
import queue
import cv2
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import logging
import argparse
import json
from typing import List, Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PythonServer")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Global parameters
class ServerConfig:
    """Server configuration class"""
    def __init__(self):
        self.buffer_size = 10  # Image queue size
        self.image_queue = queue.Queue(maxsize=self.buffer_size)
        self.reference_distribution = []
        self.negative_distribution = []
        self.counter = 0
        self.batch_counter = 0
        self.host = "localhost"
        self.port = 12345
        self.model_path = "masked_autoencoder_3d_vae.pth"
        self.discriminator_path = "discriminator_model.pth"
        self.img_size = 256
        self.patch_size = 16
        self.in_channels = 3
        self.embed_dim = 768
        self.num_layers = 12
        self.num_heads = 12
        self.latent_dim = 100
        self.out_channels = 3
        self.temporal_size = 10
        self.batch_size = 1

# Create global configuration object
config = ServerConfig()

# Model definitions
class PatchEmbedding3D(nn.Module):
    """3D image patch embedding layer"""
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super(PatchEmbedding3D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=(1, patch_size, patch_size), 
            stride=(1, patch_size, patch_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # Shape: (batch_size, embed_dim, depth, grid_size, grid_size)
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x


class MaskedViTEncoder3D(nn.Module):
    """Masked ViT encoder for 3D data"""
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, 
                 num_layers: int, num_heads: int, latent_dim: int, temporal_size: int = 10):
        super(MaskedViTEncoder3D, self).__init__()
        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches * temporal_size, embed_dim))

        # Use TransformerEncoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=4 * embed_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        self.latent_mu = nn.Linear(embed_dim, latent_dim)
        self.latent_logvar = nn.Linear(embed_dim, latent_dim)
        self.temporal_size = temporal_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Patch embedding
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, embed_dim)

        # 2. Add positional embeddings
        b, n, _ = x.size()  # x's shape is (batch_size, num_patches, embed_dim)
        x += self.pos_embedding[:, :n]

        # 3. Encode patches with TransformerEncoder
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        x_encoded = self.transformer(x)  # No tgt needed, only src

        # 4. Latent space representation
        x_encoded = x_encoded.mean(dim=0)  # (batch_size, embed_dim)
        mu = self.latent_mu(x_encoded)
        logvar = self.latent_logvar(x_encoded)

        return mu, logvar


class Decoder3D(nn.Module):
    """3D decoder"""
    def __init__(self, latent_dim: int, embed_dim: int, img_size: int, patch_size: int, 
                 out_channels: int, temporal_size: int = 10):
        super(Decoder3D, self).__init__()
        self.grid_size = img_size // patch_size
        self.linear = nn.Linear(latent_dim, embed_dim * self.grid_size * self.grid_size * temporal_size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(embed_dim // 2, embed_dim // 4, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(embed_dim // 4, out_channels, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.Sigmoid()  # Assuming output is normalized image pixels [0, 1]
        )
        self.upsample = nn.Upsample(size=(temporal_size, img_size, img_size), mode='trilinear', align_corners=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.linear(z)
        z = z.view(z.size(0), -1, 10, self.grid_size, self.grid_size)
        z = self.deconv(z)
        z = self.upsample(z)  # Upsample to match original input size
        return z


class MaskedAutoencoder3DVAE(nn.Module):
    """Masked autoencoder with 3D VAE architecture"""
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, 
                 num_layers: int, num_heads: int, latent_dim: int, out_channels: int, 
                 temporal_size: int = 10):
        super(MaskedAutoencoder3DVAE, self).__init__()
        self.encoder = MaskedViTEncoder3D(
            img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, 
            latent_dim, temporal_size
        )
        self.decoder = Decoder3D(
            latent_dim, embed_dim, img_size, patch_size, out_channels, temporal_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generate random mask with 50% masking
        # Get tensor shape
        batch_size, channels, num_frames, height, width = x.size()

        # Generate random mask with 50% masking for each frame
        mask = torch.rand(batch_size, 1, num_frames, height, width) > 0.5  # 50% probability for True/False
        mask = mask.to(x.device)  # Ensure mask is on the same device as input tensor

        # Apply mask to input tensor
        x_masked = x * mask  # Apply mask by element-wise multiplication

        # Pass input and mask to encoder
        mu, logvar = self.encoder(x_masked)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick
        # Reconstruct input
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar


class Discriminator(nn.Module):
    """Discriminator model"""
    def __init__(self):
        super(Discriminator, self).__init__()
        # First fully connected layer: input size is input_size, output size is 1024
        self.fc1 = nn.Linear(100, 1024)
        # Second fully connected layer: input size is 1024, output size is 512
        self.fc2 = nn.Linear(1024, 512)
        # Third fully connected layer: input size is 512, output size is 256
        self.fc3 = nn.Linear(512, 256)
        # Output layer: input size is 256, output size is 1
        self.fc4 = nn.Linear(256, 1)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Tanh activation function (output range is [-1, 1])
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First fully connected layer with ReLU activation
        x = self.relu(self.fc1(x))
        # Second fully connected layer with ReLU activation
        x = self.relu(self.fc2(x))
        # Third fully connected layer with ReLU activation
        x = self.relu(self.fc3(x))
        # Output layer with Tanh activation
        x = self.tanh(self.fc4(x))
        return x


class FrameDataset3D(Dataset):
    """3D frame dataset"""
    def __init__(self, folder_path: str, temporal_size: int = 10, transform: Optional[Any] = None):
        self.folder_path = folder_path
        self.temporal_size = temporal_size
        self.transform = transform
        self.frame_files = sorted([f for f in os.listdir(folder_path) if f.startswith('frame_') and f.endswith('.jpg')])
        self.num_sequences = len(self.frame_files) // temporal_size

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        frames = []
        for i in range(self.temporal_size):
            frame_path = os.path.join(self.folder_path, self.frame_files[idx * self.temporal_size + i])
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # Stack frames to create 3D tensor
        frames = torch.stack(frames, dim=0)  # Shape: (temporal_size, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # Shape: (C, temporal_size, H, W)

        return frames


# Helper functions
def re_parameterized(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick for VAE"""
    std = torch.exp(0.5 * logvar)  # Calculate standard deviation
    eps = torch.randn_like(std)    # Sample epsilon from standard normal distribution
    z = mu + eps * std             # Calculate latent vector z using reparameterization trick
    return z


def discriminator_loss(real_output: torch.Tensor, fake_output: torch.Tensor, 
                      real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
    """Discriminator loss function"""
    # Calculate first term
    real_loss = torch.mean((real_output - 1) ** 2)
    # Calculate second term
    fake_loss = torch.mean((fake_output + 1) ** 2)
    # Gradient penalty
    epsilon = torch.rand(real_samples.size(0), 1).to(device)
    epsilon = epsilon.unsqueeze(1)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad = True
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, 
        inputs=interpolated,
        grad_outputs=torch.ones(d_interpolated.size()).to(device),
        create_graph=True, 
        retain_graph=True
    )[0]
    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    # Total loss
    total_loss = real_loss + fake_loss + 10 * gradients_penalty
    return total_loss


def implicit_feature_clustering(implicit_states: torch.Tensor) -> Tuple[Dict[int, float], np.ndarray]:
    """Implicit feature clustering"""
    # Move tensor to CPU and convert to numpy
    implicit_states = implicit_states.cpu().numpy()

    # Step 1: t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    low_dim_vectors = tsne.fit_transform(implicit_states)

    # Step 2: Calculate SSE to find best k
    sse = []
    k_values = range(1, 10)  # Adjust k value range as needed
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(low_dim_vectors)
        sse.append(kmeans.inertia_)

    # Step 3: Find best k using elbow method
    sse_diff = np.abs(np.diff(sse))  # Use np.abs to calculate absolute differences
    best_k = sse_diff.argmax() + 2  # +2 because argmax gives index, we're looking at change between clusters

    logger.info(f"Best K value determined by elbow method: {best_k}")

    # Step 4: Use best k for KMeans
    best_kmeans = KMeans(n_clusters=best_k, random_state=42)
    clusters = best_kmeans.fit_predict(low_dim_vectors)

    # Step 5: Frequency calculation
    cluster_centers = best_kmeans.cluster_centers_
    cluster_labels = best_kmeans.labels_

    # Calculate frequency of each cluster
    cluster_frequencies = {i: (cluster_labels == i).sum() / len(cluster_labels) for i in range(best_k)}

    return cluster_frequencies, cluster_centers


def compute_weight_for_new_sample(new_sample: torch.Tensor, 
                                 cluster_frequencies: Dict[int, float], 
                                 cluster_centers: np.ndarray) -> float:
    """Calculate weight for new sample"""
    # Ensure new_sample is on CPU for t-SNE and convert to numpy
    new_sample = new_sample.cpu().numpy()

    # Apply t-SNE to new sample
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    new_sample_2d = tsne.fit_transform(new_sample.reshape(20, 5))

    # Calculate distances to each cluster center
    distances = np.array([np.linalg.norm(new_sample_2d - center) for center in cluster_centers])

    # Find closest cluster
    closest_cluster = distances.argmin()

    # Get weight based on frequency of closest cluster
    weight = cluster_frequencies[closest_cluster]

    return weight


# Server processing functions
def process_images(client_socket: socket.socket) -> None:
    """Process images in queue and perform model inference"""
    global config

    images = []

    # Define transformation to resize images to 256x256
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((config.img_size, config.img_size)),  # Resize to 256x256
        transforms.ToTensor()  # Convert back to tensor
    ])

    # Get all images from queue
    while not config.image_queue.empty():
        img_np = config.image_queue.get()
        img_resized = resize_transform(img_np)
        images.append(img_resized.numpy())

    # Stack images into 3D feature
    image_stack = np.stack(images, axis=0)
    image_stack_tensor = torch.from_numpy(image_stack).float()

    # Add batch dimension and permute to get desired shape
    image_stack_tensor = image_stack_tensor.unsqueeze(0)  # Shape: (1, 10, 3, 256, 256)
    image_stack_tensor = image_stack_tensor.permute(0, 2, 1, 3, 4)  # Shape: (1, 3, 10, 256, 256)

    # If model is on GPU, move tensor to GPU
    image_stack_tensor = image_stack_tensor.to(device)

    # Inference with Masked VAE
    with torch.no_grad():
        _, mu, logvar = mvae_model(image_stack_tensor)
        z = re_parameterized(mu, logvar)

    # Prepare negative distribution before training discriminator
    if config.counter >= 30 - len(config.reference_distribution):
        config.negative_distribution.append(z)

    # If condition met, start training discriminator
    if config.counter == 30:
        logger.info("Training Discriminator...")
        config.negative_distribution = torch.cat(config.negative_distribution, dim=0)
        train_discriminator()
        config.counter = 0
        config.negative_distribution = []
    config.counter += 1

    # Inference again
    with torch.no_grad():
        # Calculate probability
        # weight_for_z = compute_weight_for_new_sample(z, cluster_frequencies, cluster_centers)
        # prob = discriminator_model(z * weight_for_z)
        prob = discriminator_model(z)
        prob_np = prob.cpu().numpy()  # Move tensor to CPU and convert to Numpy array
        prob_bytes = prob_np.tobytes()
        client_socket.sendall(prob_bytes)


def handler(client_socket: socket.socket, client_address: Tuple[str, int]) -> None:
    """Thread function to handle client connection"""
    global config
    logger.info(f'Connection from client: {client_address}')
    
    while True:
        try:
            # Receive image data
            image_data = b""
            # Get length
            length_bytes = client_socket.recv(10)  # Length is 10 bytes
            length_str = length_bytes.decode('utf-8')
            length = int(length_str)  # Convert to integer
            
            # Receive complete data
            while length > 0:
                data = client_socket.recv(min(length, 4096))
                image_data += data
                length -= len(data)

            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_np is not None:
                # Successfully decoded image
                config.image_queue.put(img_np)

                # Check if 10 images have been accepted
                if config.image_queue.qsize() == 10:
                    config.batch_counter += 1
                    logger.info(f"10 images accepted! Batch: {config.batch_counter}")
                    process_images(client_socket)
                    config.image_queue.queue.clear()

        except Exception as e:
            logger.error(f"Error processing client connection: {e}")
            logger.info("Closing socket")
            client_socket.close()
            break


def train_discriminator() -> None:
    """Train discriminator model"""
    global config, discriminator_model, discriminator_optimizer

    # Create tensors from reference and negative distributions
    positive_samples = config.reference_distribution.to(device)  # Shape: (100, 1, 100)
    negative_samples = config.negative_distribution.to(device)  # Shape: (10000, 1, 100)

    # Training parameters
    num_samples = len(positive_samples)  # Number of positive samples
    num_epochs = 1000  # Training epochs
    k_batch_size = 10  # Batch size for discriminator

    for epoch in range(num_epochs):
        # Randomly sample 100 positive and 100 negative samples per batch
        for i in range(0, num_samples, k_batch_size):
            real_samples = positive_samples[i:i + k_batch_size]  # Batch of positive samples
            fake_samples = negative_samples[
                torch.randint(0, negative_samples.size(0), (k_batch_size,))]  # Random batch of negative samples

            # Ensure samples are in correct shape (batch_size, 100)
            real_samples = real_samples.view(real_samples.size(0), -1)
            fake_samples = fake_samples.view(fake_samples.size(0), -1)

            # Clear gradients
            discriminator_optimizer.zero_grad()

            # Forward pass: compute discriminator output
            real_outputs = discriminator_model(real_samples)
            fake_outputs = discriminator_model(fake_samples)

            # Compute loss using custom discriminator loss function
            loss = discriminator_loss(real_outputs, fake_outputs, real_samples, fake_samples)

            # Backward pass and optimization
            loss.backward()
            discriminator_optimizer.step()

        if (epoch + 1) % 100 == 0:
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save trained model
    torch.save(discriminator_model.state_dict(), config.discriminator_path)
    logger.info(f"Discriminator model saved to {config.discriminator_path}")


def inference_reference(ref_dataloader: DataLoader) -> torch.Tensor:
    """Inference on reference data"""
    global config, mvae_model
    
    logger.info(f'Loading model from {config.model_path}, inferencing reference z for discriminator...')

    reference_distribution = []  # Initialize reference distribution list

    # Perform inference on entire dataset
    with torch.no_grad():
        for batch in ref_dataloader:
            x = batch.to(device)

            # Pass input to model to get z output
            _, mu, logvar = mvae_model(x)
            z = re_parameterized(mu, logvar)

            # Add latent vectors to reference distribution list
            reference_distribution.append(z)

    # Stack list into single tensor
    reference_distribution = torch.cat(reference_distribution, dim=0)
    logger.info(f'Reference distribution shape: {reference_distribution.shape}')

    return reference_distribution


def load_models() -> None:
    """Load pre-trained models"""
    global mvae_model, discriminator_model, discriminator_optimizer, config
    
    # Load Masked VAE model
    mvae_model = MaskedAutoencoder3DVAE(
        config.img_size, config.patch_size, config.in_channels, config.embed_dim,
        config.num_layers, config.num_heads, config.latent_dim, config.out_channels,
        config.temporal_size
    ).to(device)
    
    if os.path.exists(config.model_path):
        mvae_model.load_state_dict(torch.load(config.model_path))
        logger.info(f"Loaded Masked VAE model: {config.model_path}")
    else:
        logger.warning(f"Masked VAE model not found: {config.model_path}, will use uninitialized model")
    
    mvae_model.eval()
    
    # Load discriminator model
    discriminator_model = Discriminator().to(device)
    
    if os.path.exists(config.discriminator_path):
        discriminator_model.load_state_dict(torch.load(config.discriminator_path))
        logger.info(f"Loaded discriminator model: {config.discriminator_path}")
    else:
        logger.warning(f"Discriminator model not found: {config.discriminator_path}, will use uninitialized model")
    
    discriminator_model.eval()
    
    # Initialize discriminator optimizer
    discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=0.0001)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Python server for communication with Unity ml-agents')
    parser.add_argument('--host', type=str, default='localhost', help='Server host address')
    parser.add_argument('--port', type=int, default=12345, help='Server port')
    parser.add_argument('--model_path', type=str, default='masked_autoencoder_3d_vae.pth', help='Masked VAE model path')
    parser.add_argument('--discriminator_path', type=str, default='discriminator_model.pth', help='Discriminator model path')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Log level')
    return parser.parse_args()


def main() -> None:
    """Main function"""
    global config, cluster_frequencies, cluster_centers
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Update configuration
    config.host = args.host
    config.port = args.port
    config.model_path = args.model_path
    config.discriminator_path = args.discriminator_path
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Load models
    load_models()
    
    # Load reference data
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
    ])
    
    ref_folder_path = r'C:\Users\10028\HKU\traj\Used\Motion Clips Look Up'
    ref_dataset = FrameDataset3D(ref_folder_path, temporal_size=config.temporal_size, transform=transform)
    ref_dataloader = DataLoader(ref_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Inference on reference data
    config.reference_distribution = inference_reference(ref_dataloader)
    cluster_frequencies, cluster_centers = implicit_feature_clustering(config.reference_distribution)
    
    # Create and listen socket
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind((config.host, config.port))
    listen_socket.listen()
    logger.info(f"Server listening for connections on {config.host}:{config.port}...")
    
    try:
        while True:
            # Accept client connection
            client_socket, client_address = listen_socket.accept()
            logger.info(f"Connection established with {client_address}")
            
            # Start new thread to handle connection
            t = threading.Thread(target=handler, args=(client_socket, client_address))
            t.daemon = True  # Set as daemon thread so it terminates when main program exits
            t.start()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        listen_socket.close()
        logger.info("Server closed")


if __name__ == '__main__':
    main()
