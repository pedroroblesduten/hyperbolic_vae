import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LoadData:
    def __init__(self, batch_size, shuffle=True, device="cuda", eta=1e-6):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.eta = eta
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.device == "cuda" else {}

    def _transform(self):
        # Define the transformations applied to each image
        tx = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda p: p.clamp(self.eta, 1 - self.eta))  # Clamping values
        ])
        return tx

    def get_train_loader(self):
        # Load the MNIST training data
        train_loader = DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=self._transform()),
            batch_size=self.batch_size, shuffle=self.shuffle, **self.kwargs
        )
        return train_loader

    def get_test_loader(self):
        # Load the MNIST test data
        test_loader = DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=self._transform()),
            batch_size=self.batch_size, shuffle=self.shuffle, **self.kwargs
        )
        return test_loader

import torch

# Example testing code for MNISTDataLoader class
if __name__ == "__main__":
    # Initialize the data loader with desired parameters
    batch_size = 64
    data_loader = LoadData(batch_size=batch_size, device="cuda")

    # Retrieve train and test loaders
    train_loader = data_loader.get_train_loader()

    # Fetch the first batch of the training data and print some info
    for images, labels in train_loader:
        print(f"Batch size: {images.size(0)}")  # Print the batch size
        print(f"Image shape: {images.shape}")   # Shape of the images
        print(f"Labels: {labels[:10]}")         # Print first 10 labels in the batch
        break  # Only process the first batch for testing
