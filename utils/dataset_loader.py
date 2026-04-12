"""
Dataset utilities for loading and managing digit recognition datasets.
Supports SVHN (Street View House Numbers) and custom datasets.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import urllib.request
import tempfile
import tarfile
from pathlib import Path
import cv2
import h5py


class SVHNDataset:
    """
    SVHN Dataset loader for digit recognition benchmarking.
    """
    
    def __init__(self, dataset_dir: str = "datasets/svhn"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.train_file = self.dataset_dir / "train_32x32.mat"
        self.test_file = self.dataset_dir / "test_32x32.mat"
        self.metadata_file = self.dataset_dir / "metadata.csv"
        
    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download SVHN dataset if not exists.
        
        Args:
            force_download: Force re-download even if files exist
            
        Returns:
            True if download successful, False otherwise
        """
        if self.train_file.exists() and self.test_file.exists() and not force_download:
            return True
            
        print("Downloading SVHN dataset...")
        
        # URLs for SVHN dataset
        urls = {
            'train': 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
            'test': 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
        }
        
        try:
            for split, url in urls.items():
                target_file = self.train_file if split == 'train' else self.test_file
                
                if target_file.exists() and not force_download:
                    continue
                    
                print(f"Downloading {split} split...")
                urllib.request.urlretrieve(url, target_file)
                print(f"Downloaded {split} to {target_file}")
                
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def load_data(self, split: str = 'test', max_samples: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load SVHN data from .mat files.
        
        Args:
            split: 'train' or 'test'
            max_samples: Maximum number of samples to load
            
        Returns:
            Tuple of (images, labels)
        """
        if not self.download_dataset():
            raise FileNotFoundError("Could not download or find SVHN dataset")
            
        file_path = self.train_file if split == 'train' else self.test_file
        
        try:
            import scipy.io as sio
            mat = sio.loadmat(file_path)
            images = mat['X']  # Shape: (32, 32, 3, N)
            labels = mat['y']  # Shape: (N, 1)
            
            # Transpose images to (N, 32, 32, 3)
            images = np.transpose(images, (3, 0, 1, 2))
            
            # Convert to uint8 and BGR for OpenCV
            images = images.astype(np.uint8)
            images = images[:, :, :, ::-1]  # RGB to BGR
            
            # Labels: SVHN uses 10 for digit 0, convert to 0-9
            labels = labels.flatten()
            labels = [str(label % 10) for label in labels]
            
            # Limit samples if specified
            if max_samples is not None and max_samples < len(images):
                images = images[:max_samples]
                labels = labels[:max_samples]
            
            return images, labels
                
        except Exception as e:
            raise RuntimeError(f"Error loading SVHN data: {e}")
    
    def get_sample_batch(self, split: str = 'test', batch_size: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Get a small batch of samples for quick testing.
        
        Args:
            split: 'train' or 'test'
            batch_size: Number of samples to return
            
        Returns:
            Tuple of (images, labels)
        """
        return self.load_data(split, max_samples=batch_size)
    
    def create_metadata(self) -> pd.DataFrame:
        """
        Create metadata CSV for the dataset.
        
        Returns:
            DataFrame with dataset metadata
        """
        if not self.download_dataset():
            return pd.DataFrame()
        
        # Load a small sample to get statistics
        sample_images, sample_labels = self.get_sample_batch('test', 100)
        
        metadata = {
            'dataset_name': ['SVHN'],
            'description': ['Street View House Numbers - Printed digits'],
            'image_size': [f"{sample_images.shape[1]}x{sample_images.shape[2]}"],
            'channels': [sample_images.shape[3]],
            'num_classes': [10],
            'train_samples': ['~73257'],
            'test_samples': ['~26032'],
            'digit_range': ['0-9'],
            'format': ['MATLAB .mat files']
        }
        
        df = pd.DataFrame(metadata)
        df.to_csv(self.metadata_file, index=False)
        return df


class CustomDigitDataset:
    """
    Custom dataset loader for user-provided digit images.
    """
    
    def __init__(self, dataset_dir: str = "datasets/custom"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
    def load_from_folder(self, folder_path: str, label_file: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load images from a folder structure.
        
        Args:
            folder_path: Path to folder containing images
            label_file: Optional CSV file with image_name, label pairs
            
        Returns:
            Tuple of (images, labels)
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder {folder_path} does not exist")
        
        images = []
        labels = []
        
        # Load labels from file if provided
        label_dict = {}
        if label_file and Path(label_file).exists():
            df = pd.read_csv(label_file)
            label_dict = dict(zip(df['image_name'], df['label']))
        
        # Load all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in folder_path.glob(ext):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        images.append(img)
                        
                        # Get label from file or dictionary
                        if img_path.name in label_dict:
                            labels.append(str(label_dict[img_path.name]))
                        else:
                            # Try to extract digits from filename
                            digits = ''.join([c for c in img_path.stem if c.isdigit()])
                            labels.append(digits if digits else "")
                            
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        if not images:
            raise ValueError("No valid images found in folder")

        # Return a plain list — avoids np.array(images) crashing when images
        # have different shapes (different resolutions in the same folder).
        return images, labels
    
    def create_sample_dataset(self, num_samples: int = 50) -> Tuple[np.ndarray, List[str]]:
        """
        Create a synthetic sample dataset for testing.
        
        Args:
            num_samples: Number of synthetic samples to create
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        for i in range(num_samples):
            # Create a simple digit image
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            digit = str(i % 10)
            
            # Add some random text
            cv2.putText(img, digit, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Add noise
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            
            images.append(img)
            labels.append(digit)
        
        return np.array(images), labels


def get_dataset_loader(dataset_type: str, **kwargs):
    """
    Factory function to get appropriate dataset loader.
    
    Args:
        dataset_type: 'svhn' or 'custom'
        **kwargs: Additional arguments for dataset loader
        
    Returns:
        Dataset loader instance
    """
    if dataset_type.lower() == 'svhn':
        return SVHNDataset(**kwargs)
    elif dataset_type.lower() == 'custom':
        return CustomDigitDataset(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def apply_image_conditions(image: np.ndarray, condition: str) -> np.ndarray:
    """
    Apply visual conditions to images for robustness testing.
    
    Args:
        image: Input image
        condition: 'clean', 'blurry', 'noisy', 'low_contrast'
        
    Returns:
        Processed image
    """
    if condition == 'clean':
        return image
    elif condition == 'blurry':
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif condition == 'noisy':
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    elif condition == 'low_contrast':
        # Reduce contrast
        image_float = image.astype(np.float32)
        image_float = (image_float - 128) * 0.5 + 128
        return np.clip(image_float, 0, 255).astype(np.uint8)
    else:
        return image


class SVHNFormat1Dataset:
    """
    SVHN Format 1 Dataset loader using our downloaded format1 dataset with separate label files.
    """
    
    def __init__(self, dataset_dir: str = "datasets/svhn_format1"):
        self.dataset_dir = Path(dataset_dir)
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"
        self.train_labels_file = self.dataset_dir / "train_labels.pkl"
        self.test_labels_file = self.dataset_dir / "test_labels.pkl"
    
    def load_data(self, split: str = 'test', max_samples: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load SVHN Format 1 data using downloaded images and label files.
        
        Args:
            split: 'train' or 'test'
            max_samples: Maximum number of samples to load
            
        Returns:
            Tuple of (images, labels)
        """
        data_dir = self.train_dir if split == 'train' else self.test_dir
        labels_file = self.train_labels_file if split == 'train' else self.test_labels_file
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # Load labels from pickle file
        import pickle
        with open(labels_file, 'rb') as f:
            label_data = pickle.load(f)
        
        labels = label_data['labels']
        
        # Load images
        image_files = sorted([f for f in data_dir.glob("*.png")])
        
        if max_samples:
            image_files = image_files[:max_samples]
            labels = labels[:max_samples]
        
        images = []
        for img_file in image_files:
            # Read image
            image = cv2.imread(str(img_file))
            if image is not None:
                # Resize to consistent size for batch processing
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
                images.append(image)
            else:
                print(f"Warning: Could not read image {img_file}")
        
        if images:
            return np.array(images), labels
        else:
            return np.array([]), labels
