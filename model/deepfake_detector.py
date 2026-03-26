"""
TrustLens - Deepfake Detection Model
Uses EfficientNet-B4 as backbone with a custom binary classification head.
Pre-trained on ImageNet, fine-tunable on FaceForensics++ or Celeb-DF datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import os
import io
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────

class EfficientNetDeepfakeDetector(nn.Module):
    """
    Deepfake detector built on EfficientNet-B4 backbone.
    Binary output: real (0) vs fake (1)
    """

    def __init__(self, dropout_rate: float = 0.4, pretrained: bool = True):
        super().__init__()

        # Load EfficientNet-B4 backbone
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)

        # Remove original classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Custom classification head for deepfake detection
        # EfficientNet-B4 outputs 1792 features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # [real, fake]
        )

        # Frequency analysis branch (catches GAN artifacts in frequency domain)
        self.freq_branch = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, freq_features=None):
        # Spatial features from EfficientNet
        x = self.features(x)
        x = self.avgpool(x)
        spatial_features = torch.flatten(x, 1)

        spatial_out = self.classifier(spatial_features)

        if freq_features is not None:
            freq_out = self.freq_branch(freq_features)
            # Ensemble: weighted average
            out = 0.7 * spatial_out + 0.3 * freq_out
        else:
            out = spatial_out

        return out

    def get_feature_embeddings(self, x):
        """Returns intermediate feature embeddings for analysis."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


# ─────────────────────────────────────────────
#  Frequency Domain Analysis
# ─────────────────────────────────────────────

def extract_frequency_features(image_np: np.ndarray) -> np.ndarray:
    """
    Extracts FFT-based frequency domain features.
    GANs and diffusion models leave characteristic artifacts in frequency space.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256))

    # Fast Fourier Transform
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1e-8)

    # Normalize
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

    # Flatten to feature vector (center 32x32 captures low-freq patterns)
    h, w = magnitude.shape
    center = magnitude[h//2-16:h//2+16, w//2-16:w//2+16]
    features = center.flatten()  # 1024 features

    return features.astype(np.float32)


# ─────────────────────────────────────────────
#  Face Extractor (OpenCV DNN — no extra deps)
# ─────────────────────────────────────────────

class FaceExtractor:
    """
    Extracts faces from images using OpenCV DNN face detector.
    Falls back to full image if no face is detected.
    """

    def __init__(self):
        self.detector = None
        self._load_detector()

    def _load_detector(self):
        """Load OpenCV's DNN face detector."""
        try:
            # Use OpenCV's built-in Haar cascade as fallback
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("Face detector loaded (Haar Cascade)")
        except Exception as e:
            logger.warning(f"Face detector load failed: {e}. Will use full image.")
            self.detector = None

    def extract_face(self, image_np: np.ndarray, margin: float = 0.3):
        """
        Detects and crops the largest face with margin.
        Returns cropped face or full image if no face found.
        """
        if self.detector is None:
            return image_np

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) == 0:
            logger.debug("No face detected, using full image")
            return image_np

        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        H, W = image_np.shape[:2]

        # Add margin
        mx = int(w * margin)
        my = int(h * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(W, x + w + mx)
        y2 = min(H, y + h + my)

        face_crop = image_np[y1:y2, x1:x2]
        return face_crop if face_crop.size > 0 else image_np


# ─────────────────────────────────────────────
#  Inference Pipeline
# ─────────────────────────────────────────────

class DeepfakeDetectionPipeline:
    """
    Full inference pipeline:
    Image/Video → Face Detection → Preprocessing → EfficientNet → Result
    """

    # ImageNet normalization (EfficientNet-B4)
    TRANSFORM = transforms.Compose([
        transforms.Resize((380, 380)),  # EfficientNet-B4 native resolution
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.model = EfficientNetDeepfakeDetector(pretrained=True)
        self.model = self.model.to(self.device)

        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
            logger.info(f"Loaded fine-tuned weights from {model_path}")
        else:
            logger.info("Using ImageNet pre-trained weights (no fine-tuned checkpoint found)")
            logger.info("→ For best accuracy, fine-tune on FaceForensics++ dataset")

        self.model.eval()
        self.face_extractor = FaceExtractor()

    def _load_weights(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def _preprocess_image(self, image_np: np.ndarray) -> tuple:
        """Face extraction + tensor preprocessing."""
        face = self.face_extractor.extract_face(image_np)
        freq_features = extract_frequency_features(face)

        pil_img = Image.fromarray(face).convert('RGB')
        tensor = self.TRANSFORM(pil_img).unsqueeze(0).to(self.device)
        freq_tensor = torch.tensor(freq_features).unsqueeze(0).to(self.device)

        return tensor, freq_tensor

    def predict_image(self, image_data: bytes) -> dict:
        """
        Predict deepfake probability for a single image.

        Returns:
            dict with keys: label, confidence, fake_prob, real_prob, face_detected
        """
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_np = np.array(image)

            tensor, freq_tensor = self._preprocess_image(image_np)

            with torch.no_grad():
                logits = self.model(tensor, freq_tensor)
                probs = F.softmax(logits, dim=1)
                fake_prob = probs[0][1].item()
                real_prob = probs[0][0].item()

            label = 'FAKE' if fake_prob > 0.5 else 'REAL'
            confidence = max(fake_prob, real_prob) * 100

            return {
                'success': True,
                'label': label,
                'confidence': round(confidence, 2),
                'fake_probability': round(fake_prob * 100, 2),
                'real_probability': round(real_prob * 100, 2),
                'face_detected': True,
                'analysis_method': 'EfficientNet-B4 + Frequency Analysis'
            }

        except Exception as e:
            logger.error(f"Image prediction error: {e}")
            return {'success': False, 'error': str(e)}

    def predict_video(self, video_path: str, sample_frames: int = 15) -> dict:
        """
        Predict deepfake probability for a video by sampling frames.

        Args:
            video_path: Path to video file
            sample_frames: Number of frames to analyze

        Returns:
            Aggregated prediction across sampled frames
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0:
                return {'success': False, 'error': 'Could not read video'}

            # Sample frames evenly across the video
            frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
            frame_predictions = []
            frames_analyzed = 0

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    tensor, freq_tensor = self._preprocess_image(frame_rgb)
                    with torch.no_grad():
                        logits = self.model(tensor, freq_tensor)
                        probs = F.softmax(logits, dim=1)
                        frame_predictions.append(probs[0][1].item())
                        frames_analyzed += 1
                except Exception:
                    continue

            cap.release()

            if not frame_predictions:
                return {'success': False, 'error': 'No frames could be analyzed'}

            fake_prob = float(np.mean(frame_predictions))
            real_prob = 1.0 - fake_prob
            label = 'FAKE' if fake_prob > 0.5 else 'REAL'
            confidence = max(fake_prob, real_prob) * 100

            return {
                'success': True,
                'label': label,
                'confidence': round(confidence, 2),
                'fake_probability': round(fake_prob * 100, 2),
                'real_probability': round(real_prob * 100, 2),
                'frames_analyzed': frames_analyzed,
                'total_frames': total_frames,
                'duration_seconds': round(total_frames / fps, 1) if fps > 0 else 0,
                'analysis_method': 'EfficientNet-B4 + Temporal Aggregation'
            }

        except Exception as e:
            logger.error(f"Video prediction error: {e}")
            return {'success': False, 'error': str(e)}
