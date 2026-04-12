"""
SVHN-specific digit classifier.

The cropped 32x32 SVHN dataset is a single-digit classification task, not a
multi-digit detection task. This classifier provides a fallback when OCR
pipelines fail to produce a valid single-digit prediction.
"""

import cv2
import numpy as np

from .dataset_loader import SVHNDataset


class SVHNDigitClassifier:
    """Lightweight HOG + k-NN classifier for SVHN crops."""

    def __init__(self, training_samples: int = 10000, k: int = 5):
        self.training_samples = training_samples
        self.k = k
        self.knn = cv2.ml.KNearest_create()
        self.knn.setDefaultK(k)
        self.hog = cv2.HOGDescriptor(
            (32, 32),
            (16, 16),
            (8, 8),
            (8, 8),
            9,
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.is_trained = False

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image to the 32x32 grayscale format used by the classifier."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if gray.shape[:2] != (32, 32):
            interpolation = cv2.INTER_AREA if gray.shape[0] > 32 or gray.shape[1] > 32 else cv2.INTER_CUBIC
            gray = cv2.resize(gray, (32, 32), interpolation=interpolation)

        return self.clahe.apply(gray)

    def _compute_features(self, image: np.ndarray) -> np.ndarray:
        """Compute HOG features for a single image."""
        prepared = self._prepare_image(image)
        return self.hog.compute(prepared).reshape(1, -1).astype(np.float32)

    def train(self, dataset: SVHNDataset | None = None):
        """Train the classifier on the SVHN training split."""
        dataset = dataset or SVHNDataset()
        images, labels = dataset.load_data("train", max_samples=self.training_samples)

        features = np.vstack([self._compute_features(image) for image in images])
        responses = np.array(labels, dtype=np.float32)

        self.knn.train(features, cv2.ml.ROW_SAMPLE, responses)
        self.is_trained = True
        return self

    def predict(self, image: np.ndarray) -> str:
        """Predict the digit in a single SVHN image."""
        if not self.is_trained:
            raise RuntimeError("SVHN classifier is not trained.")

        features = self._compute_features(image)
        _, results, _, _ = self.knn.findNearest(features, self.k)
        return str(int(results[0][0]))

    def predict_batch(self, images: list[np.ndarray]) -> list[str]:
        """Predict digits for a batch of images."""
        return [self.predict(image) for image in images]
