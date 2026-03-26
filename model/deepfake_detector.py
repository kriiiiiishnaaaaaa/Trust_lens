"""
TrustLens - Deepfake Detection via Groq Vision AI
Replaces the untrained EfficientNet with Groq's vision model for accurate detection.
"""

import os
import io
import cv2
import json
import base64
import logging
import numpy as np
from PIL import Image
from groq import Groq

logger = logging.getLogger(__name__)

GROQ_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

DEEPFAKE_PROMPT = """You are a forensic media analyst specializing in deepfake and AI-generated face detection.

Carefully examine this image for signs of manipulation or AI generation. Check for:
1. Facial boundary artifacts — blurring, blending seams, unnatural edges around the face
2. Skin texture anomalies — over-smoothing, missing pores, plastic appearance
3. Eye inconsistencies — unnatural reflections, misaligned gaze, wrong iris details
4. Lighting mismatches — shadows that don't match light sources, unnatural specular highlights
5. GAN/diffusion artifacts — checkerboard patterns, frequency noise, asymmetric features
6. Background–face inconsistency — mismatched sharpness or compression

Respond with ONLY a valid JSON object, no explanation outside the JSON:
{
  "label": "FAKE" or "REAL",
  "fake_probability": <integer 0-100>,
  "real_probability": <integer 0-100>,
  "confidence": <integer 0-100>,
  "face_detected": true or false,
  "reasoning": "<one sentence summary>",
  "artifacts_found": ["artifact1", "artifact2"]
}"""


class DeepfakeDetectionPipeline:
    """
    Inference pipeline powered by Groq Vision API.
    Same public interface as the original PyTorch pipeline.
    """

    def __init__(self, model_path: str = None, device: str = None):
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        self.client = Groq(api_key=api_key)
        self.model = os.getenv('GROQ_MODEL', GROQ_DEFAULT_MODEL)
        logger.info(f"Groq Vision pipeline ready — model: {self.model}")

    # ─────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────

    def _to_b64_jpeg(self, image_bytes: bytes, max_dim: int = 800) -> str:
        """Resize + compress image and return base64 JPEG string."""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _frame_to_b64_jpeg(self, frame_rgb: np.ndarray, max_dim: int = 640) -> str:
        """Convert a numpy frame (RGB) to base64 JPEG."""
        img = Image.fromarray(frame_rgb)
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _call_groq(self, image_b64: str) -> dict:
        """Send image to Groq vision model and parse structured JSON response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        },
                        {
                            "type": "text",
                            "text": DEEPFAKE_PROMPT
                        }
                    ]
                }
            ],
            max_tokens=512,
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if '```' in content:
            parts = content.split('```')
            for part in parts:
                stripped = part.strip()
                if stripped.startswith('json'):
                    stripped = stripped[4:].strip()
                if stripped.startswith('{'):
                    content = stripped
                    break

        return json.loads(content)

    # ─────────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────────

    def predict_image(self, image_data: bytes) -> dict:
        """
        Analyze a single image for deepfake indicators.
        Returns dict with keys: success, label, confidence, fake_probability,
                                real_probability, face_detected, reasoning, artifacts_found
        """
        try:
            image_b64 = self._to_b64_jpeg(image_data)
            result = self._call_groq(image_b64)

            fake_prob = float(result.get('fake_probability', 50))
            real_prob = float(result.get('real_probability', 100 - fake_prob))

            return {
                'success': True,
                'label': result.get('label', 'FAKE' if fake_prob >= 50 else 'REAL'),
                'confidence': round(float(result.get('confidence', max(fake_prob, real_prob))), 2),
                'fake_probability': round(fake_prob, 2),
                'real_probability': round(real_prob, 2),
                'face_detected': result.get('face_detected', True),
                'reasoning': result.get('reasoning', ''),
                'artifacts_found': result.get('artifacts_found', []),
                'analysis_method': f'Groq Vision AI ({self.model})'
            }

        except Exception as e:
            logger.error(f"Image prediction error: {e}")
            return {'success': False, 'error': str(e)}

    def predict_video(self, video_path: str, sample_frames: int = 8) -> dict:
        """
        Analyze a video by sampling frames and aggregating results.
        Returns aggregated prediction across sampled frames.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0:
                return {'success': False, 'error': 'Could not read video'}

            n = min(sample_frames, total_frames)
            frame_indices = np.linspace(0, total_frames - 1, n, dtype=int)
            fake_probs = []
            frames_analyzed = 0

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    image_b64 = self._frame_to_b64_jpeg(frame_rgb)
                    result = self._call_groq(image_b64)
                    fake_probs.append(float(result.get('fake_probability', 50)) / 100.0)
                    frames_analyzed += 1
                except Exception as frame_err:
                    logger.warning(f"Frame {idx} skipped: {frame_err}")
                    continue

            cap.release()

            if not fake_probs:
                return {'success': False, 'error': 'No frames could be analyzed'}

            fake_prob = float(np.mean(fake_probs))
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
                'analysis_method': f'Groq Vision AI ({self.model}) + Temporal Aggregation'
            }

        except Exception as e:
            logger.error(f"Video prediction error: {e}")
            return {'success': False, 'error': str(e)}
