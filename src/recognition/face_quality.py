"""
Face Quality Assessment Module for Team 1
Evaluates face image quality for improved recognition accuracy
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import math

logger = logging.getLogger(__name__)


class FaceQualityAssessor:
    """
    Comprehensive face quality assessment for improved recognition accuracy.
    
    Evaluates:
    - Image resolution and face size
    - Lighting conditions and contrast
    - Face pose and orientation
    - Blur and noise levels
    - Occlusion detection
    - Expression and eye openness
    """
    
    def __init__(self, config: Dict = None):
        """Initialize face quality assessor."""
        self.config = config or {}
        
        # Quality thresholds
        self.min_face_size = self.config.get('min_face_size', 80)
        self.min_resolution = self.config.get('min_resolution', 64)
        self.min_contrast = self.config.get('min_contrast', 30)
        self.max_blur = self.config.get('max_blur', 100)
        self.min_eye_openness = self.config.get('min_eye_openness', 0.3)
        
        # Load face landmark detector
        self.landmark_detector = self._load_landmark_detector()
        
        # Quality metrics weights
        self.weights = {
            'resolution': 0.25,
            'lighting': 0.20,
            'pose': 0.20,
            'blur': 0.15,
            'occlusion': 0.10,
            'expression': 0.10
        }
    
    def _load_landmark_detector(self):
        """Load face landmark detector for pose and expression analysis."""
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            detector = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe face mesh detector loaded successfully")
            return {'type': 'mediapipe', 'detector': detector}
        except ImportError:
            logger.warning("MediaPipe not available, using basic quality assessment")
            return {'type': 'basic', 'detector': None}
        except Exception as e:
            logger.warning(f"Failed to load landmark detector: {e}")
            return {'type': 'basic', 'detector': None}
    
    def assess_face_quality(self, face_image: np.ndarray, face_bbox: List[int] = None) -> Dict:
        """
        Assess overall quality of a face image.
        
        Args:
            face_image: Face image (cropped or full frame)
            face_bbox: Bounding box [x1, y1, x2, y2] if face_image is full frame
            
        Returns:
            Dictionary containing quality scores and recommendations
        """
        if face_image is None or face_image.size == 0:
            return self._get_empty_quality_result()
        
        # Crop face region if bbox provided
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_image = face_image[y1:y2, x1:x2]
        
        # Ensure face image is valid
        if face_image.size == 0:
            return self._get_empty_quality_result()
        
        # Calculate individual quality metrics
        quality_metrics = {
            'resolution': self._assess_resolution_quality(face_image),
            'lighting': self._assess_lighting_quality(face_image),
            'pose': self._assess_pose_quality(face_image),
            'blur': self._assess_blur_quality(face_image),
            'occlusion': self._assess_occlusion_quality(face_image),
            'expression': self._assess_expression_quality(face_image)
        }
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(quality_metrics)
        
        # Generate quality assessment result
        result = {
            'overall_score': overall_score,
            'quality_level': self._get_quality_level(overall_score),
            'metrics': quality_metrics,
            'recommendations': self._generate_recommendations(quality_metrics),
            'is_suitable_for_recognition': overall_score >= 0.7
        }
        
        return result
    
    def _assess_resolution_quality(self, face_image: np.ndarray) -> Dict:
        """Assess face image resolution and size quality."""
        height, width = face_image.shape[:2]
        
        # Calculate face size
        face_size = min(width, height)
        
        # Resolution quality score (0-1)
        if face_size >= self.min_face_size:
            resolution_score = min(1.0, face_size / 200)  # Normalize to 200px
        else:
            resolution_score = max(0.0, face_size / self.min_face_size)
        
        # Check minimum resolution requirement
        meets_min_resolution = face_size >= self.min_resolution
        
        return {
            'score': resolution_score,
            'face_size': face_size,
            'width': width,
            'height': height,
            'meets_min_resolution': meets_min_resolution,
            'issues': [] if meets_min_resolution else ['Face too small for reliable recognition']
        }
    
    def _assess_lighting_quality(self, face_image: np.ndarray) -> Dict:
        """Assess lighting conditions and contrast quality."""
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Lighting quality score (0-1)
        contrast_score = min(1.0, contrast / 50)  # Normalize to contrast of 50
        
        # Brightness score (penalize too dark or too bright)
        if 30 <= brightness <= 225:
            brightness_score = 1.0
        elif brightness < 30:
            brightness_score = max(0.0, brightness / 30)
        else:
            brightness_score = max(0.0, (255 - brightness) / 30)
        
        # Combined lighting score
        lighting_score = (contrast_score + brightness_score) / 2
        
        # Identify lighting issues
        issues = []
        if contrast < self.min_contrast:
            issues.append('Low contrast - poor lighting conditions')
        if brightness < 30:
            issues.append('Too dark - insufficient lighting')
        elif brightness > 225:
            issues.append('Too bright - overexposed')
        
        return {
            'score': lighting_score,
            'contrast': contrast,
            'brightness': brightness,
            'contrast_score': contrast_score,
            'brightness_score': brightness_score,
            'issues': issues
        }
    
    def _assess_pose_quality(self, face_image: np.ndarray) -> Dict:
        """Assess face pose and orientation quality."""
        if self.landmark_detector['type'] == 'mediapipe':
            return self._assess_pose_mediapipe(face_image)
        else:
            return self._assess_pose_basic(face_image)
    
    def _assess_pose_mediapipe(self, face_image: np.ndarray) -> Dict:
        """Assess pose using MediaPipe face mesh."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Detect landmarks
            results = self.landmark_detector['detector'].process(rgb_image)
            
            if not results.multi_face_landmarks:
                return self._get_default_pose_result()
            
            landmarks = results.multi_face_landmarks[0]
            
            # Calculate pose angles
            pose_angles = self._calculate_pose_angles(landmarks, face_image.shape)
            
            # Pose quality score based on angles
            yaw_score = max(0.0, 1.0 - abs(pose_angles['yaw']) / 45)  # ±45° tolerance
            pitch_score = max(0.0, 1.0 - abs(pose_angles['pitch']) / 30)  # ±30° tolerance
            roll_score = max(0.0, 1.0 - abs(pose_angles['roll']) / 20)  # ±20° tolerance
            
            pose_score = (yaw_score + pitch_score + roll_score) / 3
            
            # Identify pose issues
            issues = []
            if abs(pose_angles['yaw']) > 30:
                issues.append('Face turned too much (yaw)')
            if abs(pose_angles['pitch']) > 20:
                issues.append('Face tilted too much (pitch)')
            if abs(pose_angles['roll']) > 15:
                issues.append('Face rotated too much (roll)')
            
            return {
                'score': pose_score,
                'yaw': pose_angles['yaw'],
                'pitch': pose_angles['pitch'],
                'roll': pose_angles['roll'],
                'yaw_score': yaw_score,
                'pitch_score': pitch_score,
                'roll_score': roll_score,
                'issues': issues
            }
            
        except Exception as e:
            logger.warning(f"MediaPipe pose assessment failed: {e}")
            return self._assess_pose_basic(face_image)
    
    def _assess_pose_basic(self, face_image: np.ndarray) -> Dict:
        """Basic pose assessment using image properties."""
        height, width = face_image.shape[:2]
        
        # Simple symmetry-based pose assessment
        center_x = width // 2
        
        # Split image into left and right halves
        left_half = face_image[:, :center_x]
        right_half = face_image[:, center_x:2*center_x]
        
        if left_half.shape != right_half.shape:
            return self._get_default_pose_result()
        
        # Flip right half for comparison
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Calculate symmetry
        diff = cv2.absdiff(left_half, right_half_flipped)
        symmetry_score = 1.0 - (np.mean(diff) / 255)
        
        # Basic pose score
        pose_score = max(0.3, symmetry_score * 0.8)  # Minimum score for basic assessment
        
        issues = []
        if symmetry_score < 0.6:
            issues.append('Face not frontal - poor symmetry')
        
        return {
            'score': pose_score,
            'yaw': 0.0,  # Unknown
            'pitch': 0.0,  # Unknown
            'roll': 0.0,  # Unknown
            'symmetry_score': symmetry_score,
            'issues': issues
        }
    
    def _calculate_pose_angles(self, landmarks, image_shape) -> Dict:
        """Calculate pose angles from MediaPipe landmarks."""
        height, width = image_shape[:2]
        
        # Key landmark indices for pose calculation
        # These are approximate - in practice you'd use more sophisticated methods
        nose_tip = landmarks.landmark[1]  # Nose tip
        left_eye = landmarks.landmark[33]  # Left eye center
        right_eye = landmarks.landmark[263]  # Right eye center
        left_ear = landmarks.landmark[234]  # Left ear
        right_ear = landmarks.landmark[454]  # Right ear
        
        # Convert to pixel coordinates
        nose_x, nose_y = int(nose_tip.x * width), int(nose_tip.y * height)
        left_eye_x, left_eye_y = int(left_eye.x * width), int(left_eye.y * height)
        right_eye_x, right_eye_y = int(right_eye.x * width), int(right_eye.y * height)
        left_ear_x, left_ear_y = int(left_ear.x * width), int(left_ear.y * height)
        right_ear_x, right_ear_y = int(right_ear.x * width), int(right_ear.y * height)
        
        # Calculate yaw (left-right rotation)
        eye_center_x = (left_eye_x + right_eye_x) / 2
        yaw = math.atan2(nose_x - eye_center_x, width / 2) * 180 / math.pi
        
        # Calculate pitch (up-down rotation)
        eye_center_y = (left_eye_y + right_eye_y) / 2
        pitch = math.atan2(nose_y - eye_center_y, height / 2) * 180 / math.pi
        
        # Calculate roll (tilt)
        roll = math.atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / math.pi
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }
    
    def _assess_blur_quality(self, face_image: np.ndarray) -> Dict:
        """Assess image blur and sharpness quality."""
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Calculate Laplacian variance (measure of sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Blur quality score (0-1)
        if sharpness > self.max_blur:
            blur_score = 1.0
        else:
            blur_score = max(0.0, sharpness / self.max_blur)
        
        # Identify blur issues
        issues = []
        if sharpness < 50:
            issues.append('Image too blurry - poor focus')
        elif sharpness < 100:
            issues.append('Image slightly blurry - may affect recognition')
        
        return {
            'score': blur_score,
            'sharpness': sharpness,
            'issues': issues
        }
    
    def _assess_occlusion_quality(self, face_image: np.ndarray) -> Dict:
        """Assess face occlusion quality."""
        if self.landmark_detector['type'] == 'mediapipe':
            return self._assess_occlusion_mediapipe(face_image)
        else:
            return self._assess_occlusion_basic(face_image)
    
    def _assess_occlusion_mediapipe(self, face_image: np.ndarray) -> Dict:
        """Assess occlusion using MediaPipe landmarks."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Detect landmarks
            results = self.landmark_detector['detector'].process(rgb_image)
            
            if not results.multi_face_landmarks:
                return self._get_default_occlusion_result()
            
            landmarks = results.multi_face_landmarks[0]
            
            # Check key facial regions for occlusion
            occlusion_scores = {
                'eyes': self._check_eye_occlusion(landmarks, face_image.shape),
                'nose': self._check_nose_occlusion(landmarks, face_image.shape),
                'mouth': self._check_mouth_occlusion(landmarks, face_image.shape)
            }
            
            # Overall occlusion score
            occlusion_score = np.mean(list(occlusion_scores.values()))
            
            # Identify occlusion issues
            issues = []
            for region, score in occlusion_scores.items():
                if score < 0.7:
                    issues.append(f'{region.capitalize()} partially occluded')
            
            return {
                'score': occlusion_score,
                'region_scores': occlusion_scores,
                'issues': issues
            }
            
        except Exception as e:
            logger.warning(f"MediaPipe occlusion assessment failed: {e}")
            return self._assess_occlusion_basic(face_image)
    
    def _assess_occlusion_basic(self, face_image: np.ndarray) -> Dict:
        """Basic occlusion assessment using image analysis."""
        # Simple edge density analysis
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Basic occlusion score
        occlusion_score = min(1.0, edge_density / 0.1)  # Normalize to expected edge density
        
        issues = []
        if edge_density < 0.05:
            issues.append('Low edge density - possible occlusion or poor quality')
        
        return {
            'score': occlusion_score,
            'edge_density': edge_density,
            'issues': issues
        }
    
    def _check_eye_occlusion(self, landmarks, image_shape) -> float:
        """Check if eyes are occluded."""
        # Simplified eye occlusion check
        # In practice, you'd implement more sophisticated occlusion detection
        return 0.9  # Placeholder
    
    def _check_nose_occlusion(self, landmarks, image_shape) -> float:
        """Check if nose is occluded."""
        return 0.9  # Placeholder
    
    def _check_mouth_occlusion(self, landmarks, image_shape) -> float:
        """Check if mouth is occluded."""
        return 0.9  # Placeholder
    
    def _assess_expression_quality(self, face_image: np.ndarray) -> Dict:
        """Assess facial expression quality for recognition."""
        if self.landmark_detector['type'] == 'mediapipe':
            return self._assess_expression_mediapipe(face_image)
        else:
            return self._assess_expression_basic(face_image)
    
    def _assess_expression_mediapipe(self, face_image: np.ndarray) -> Dict:
        """Assess expression using MediaPipe landmarks."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Detect landmarks
            results = self.landmark_detector['detector'].process(rgb_image)
            
            if not results.multi_face_landmarks:
                return self._get_default_expression_result()
            
            landmarks = results.multi_face_landmarks[0]
            
            # Check eye openness
            eye_openness = self._calculate_eye_openness(landmarks, face_image.shape)
            
            # Expression quality score
            expression_score = eye_openness
            
            issues = []
            if eye_openness < self.min_eye_openness:
                issues.append('Eyes too closed - poor for recognition')
            
            return {
                'score': expression_score,
                'eye_openness': eye_openness,
                'issues': issues
            }
            
        except Exception as e:
            logger.warning(f"MediaPipe expression assessment failed: {e}")
            return self._assess_expression_basic(face_image)
    
    def _assess_expression_basic(self, face_image: np.ndarray) -> Dict:
        """Basic expression assessment."""
        # Simple brightness-based assessment
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Basic expression score
        expression_score = 0.8  # Default score for basic assessment
        
        return {
            'score': expression_score,
            'eye_openness': 0.8,  # Unknown
            'issues': []
        }
    
    def _calculate_eye_openness(self, landmarks, image_shape) -> float:
        """Calculate eye openness ratio."""
        # Simplified eye openness calculation
        # In practice, you'd measure the ratio of eye height to width
        return 0.8  # Placeholder
    
    def _calculate_overall_score(self, quality_metrics: Dict) -> float:
        """Calculate overall quality score from individual metrics."""
        overall_score = 0.0
        
        for metric_name, weight in self.weights.items():
            if metric_name in quality_metrics:
                metric_score = quality_metrics[metric_name].get('score', 0.0)
                overall_score += weight * metric_score
        
        return round(overall_score, 3)
    
    def _get_quality_level(self, overall_score: float) -> str:
        """Get quality level description."""
        if overall_score >= 0.9:
            return "Excellent"
        elif overall_score >= 0.8:
            return "Very Good"
        elif overall_score >= 0.7:
            return "Good"
        elif overall_score >= 0.6:
            return "Fair"
        elif overall_score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def _generate_recommendations(self, quality_metrics: Dict) -> List[str]:
        """Generate improvement recommendations based on quality metrics."""
        recommendations = []
        
        for metric_name, metric_data in quality_metrics.items():
            if 'issues' in metric_data and metric_data['issues']:
                for issue in metric_data['issues']:
                    recommendations.append(f"{metric_name.capitalize()}: {issue}")
        
        if not recommendations:
            recommendations.append("Image quality is suitable for face recognition")
        
        return recommendations
    
    def _get_empty_quality_result(self) -> Dict:
        """Return empty quality result for invalid inputs."""
        return {
            'overall_score': 0.0,
            'quality_level': 'Invalid',
            'metrics': {},
            'recommendations': ['Invalid or empty image provided'],
            'is_suitable_for_recognition': False
        }
    
    def _get_default_pose_result(self) -> Dict:
        """Return default pose assessment result."""
        return {
            'score': 0.5,
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'issues': ['Pose assessment not available']
        }
    
    def _get_default_occlusion_result(self) -> Dict:
        """Return default occlusion assessment result."""
        return {
            'score': 0.5,
            'region_scores': {},
            'issues': ['Occlusion assessment not available']
        }
    
    def _get_default_expression_result(self) -> Dict:
        """Return default expression assessment result."""
        return {
            'score': 0.5,
            'eye_openness': 0.5,
            'issues': ['Expression assessment not available']
        }
    
    def batch_assess_quality(self, face_images: List[np.ndarray], 
                           face_bboxes: List[List[int]] = None) -> List[Dict]:
        """Assess quality for multiple face images."""
        results = []
        
        for i, face_image in enumerate(face_images):
            bbox = face_bboxes[i] if face_bboxes and i < len(face_bboxes) else None
            result = self.assess_face_quality(face_image, bbox)
            results.append(result)
        
        return results
    
    def filter_by_quality(self, face_images: List[np.ndarray], 
                         face_bboxes: List[List[int]] = None,
                         min_score: float = 0.7) -> Tuple[List[np.ndarray], List[List[int]], List[int]]:
        """Filter face images by quality score."""
        if face_bboxes is None:
            face_bboxes = [None] * len(face_images)
        
        # Assess quality for all images
        quality_results = self.batch_assess_quality(face_images, face_bboxes)
        
        # Filter by quality score
        filtered_images = []
        filtered_bboxes = []
        filtered_indices = []
        
        for i, (image, bbox, quality_result) in enumerate(zip(face_images, face_bboxes, quality_results)):
            if quality_result['overall_score'] >= min_score:
                filtered_images.append(image)
                filtered_bboxes.append(bbox)
                filtered_indices.append(i)
        
        return filtered_images, filtered_bboxes, filtered_indices


def create_quality_assessor(config: Dict = None) -> FaceQualityAssessor:
    """Factory function to create face quality assessor."""
    return FaceQualityAssessor(config)


if __name__ == "__main__":
    # Example usage
    config = {
        'min_face_size': 80,
        'min_resolution': 64,
        'min_contrast': 30,
        'max_blur': 100,
        'min_eye_openness': 0.3
    }
    
    assessor = create_quality_assessor(config)
    print("Face Quality Assessor created successfully!")
