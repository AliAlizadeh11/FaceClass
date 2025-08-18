"""
Layout analysis module for FaceClass project.
Generates heatmaps and analyzes spatial distribution of faces in classroom.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LayoutMapper:
    """Layout analysis and heatmap generation for classroom analysis."""
    
    def __init__(self, config):
        """Initialize layout mapper with configuration."""
        self.config = config
        self.heatmap_resolution = config.get('heatmap.resolution', (100, 100))
        self.blur_radius = config.get('heatmap.blur_radius', 5)
        self.color_map = config.get('heatmap.color_map', 'hot')
        
        # Classroom layout settings
        self.classroom_width = 1920  # pixels
        self.classroom_height = 1080  # pixels
        
        # Seat mapping
        self.seat_positions = self._generate_seat_positions()
        
        # Analysis results
        self.heatmaps = {}
        self.seat_assignments = {}
    
    def _generate_seat_positions(self) -> Dict[str, Tuple[int, int]]:
        """Generate typical classroom seat positions."""
        seats = {}
        rows = 5
        cols = 8
        
        # Calculate seat spacing
        seat_width = self.classroom_width // (cols + 1)
        seat_height = self.classroom_height // (rows + 1)
        
        for row in range(rows):
            for col in range(cols):
                seat_id = f"R{row+1}C{col+1}"
                x = (col + 1) * seat_width
                y = (row + 1) * seat_height
                seats[seat_id] = (x, y)
        
        return seats
    
    def generate_heatmap(self, detections: List[Dict], 
                        heatmap_type: str = 'presence') -> np.ndarray:
        """Generate heatmap from face detections."""
        if not detections:
            logger.warning("No detections provided for heatmap generation")
            return np.zeros(self.heatmap_resolution)
        
        # Initialize heatmap
        heatmap = np.zeros(self.heatmap_resolution)
        
        # Map detections to heatmap
        for detection in detections:
            if 'bbox' in detection:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Normalize to heatmap coordinates
                heatmap_x = int((center_x / self.classroom_width) * self.heatmap_resolution[0])
                heatmap_y = int((center_y / self.classroom_height) * self.heatmap_resolution[1])
                
                # Ensure coordinates are within bounds
                heatmap_x = max(0, min(self.heatmap_resolution[0] - 1, heatmap_x))
                heatmap_y = max(0, min(self.heatmap_resolution[1] - 1, heatmap_y))
                
                # Add value based on heatmap type
                if heatmap_type == 'presence':
                    heatmap[heatmap_y, heatmap_x] += 1
                elif heatmap_type == 'attention':
                    attention_score = detection.get('attention', {}).get('attention_score', 0.5)
                    heatmap[heatmap_y, heatmap_x] += attention_score
                elif heatmap_type == 'emotion':
                    emotion_score = detection.get('emotion_confidence', 0.5)
                    heatmap[heatmap_y, heatmap_x] += emotion_score
        
        # Apply Gaussian blur for smooth visualization
        if self.blur_radius > 0:
            heatmap = cv2.GaussianBlur(heatmap, (self.blur_radius, self.blur_radius), 0)
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def generate_multiple_heatmaps(self, detections: List[Dict]) -> Dict[str, np.ndarray]:
        """Generate multiple types of heatmaps."""
        heatmap_types = ['presence', 'attention', 'emotion']
        heatmaps = {}
        
        for heatmap_type in heatmap_types:
            heatmaps[heatmap_type] = self.generate_heatmap(detections, heatmap_type)
        
        self.heatmaps = heatmaps
        return heatmaps
    
    def assign_seats(self, detections: List[Dict]) -> Dict[str, Dict]:
        """Assign detected faces to nearest seats."""
        seat_assignments = {}
        
        # Group detections by track_id to avoid duplicates
        track_groups = {}
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id not in track_groups:
                track_groups[track_id] = detection
        
        # Assign each track to nearest seat
        used_seats = set()
        for track_id, detection in track_groups.items():
            if 'bbox' in detection:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Find nearest available seat
                nearest_seat = None
                min_distance = float('inf')
                
                for seat_id, (seat_x, seat_y) in self.seat_positions.items():
                    if seat_id in used_seats:
                        continue
                    
                    distance = np.sqrt((center_x - seat_x)**2 + (center_y - seat_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_seat = seat_id
                
                if nearest_seat:
                    used_seats.add(nearest_seat)
                    seat_assignments[nearest_seat] = {
                        'track_id': track_id,
                        'identity': detection.get('identity', 'Unknown'),
                        'attention_score': detection.get('attention', {}).get('attention_score', 0),
                        'dominant_emotion': detection.get('dominant_emotion', 'neutral'),
                        'position': (center_x, center_y)
                    }
        
        self.seat_assignments = seat_assignments
        return seat_assignments
    
    def visualize_heatmap(self, heatmap: np.ndarray, title: str = "Heatmap", 
                         save_path: Optional[str] = None) -> None:
        """Visualize heatmap using matplotlib."""
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap=self.color_map, interpolation='bilinear')
        plt.colorbar(label='Intensity')
        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to: {save_path}")
        
        plt.show()
    
    def visualize_classroom_layout(self, detections: List[Dict], 
                                 save_path: Optional[str] = None) -> None:
        """Visualize classroom layout with seat assignments."""
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw classroom boundaries
        ax.add_patch(plt.Rectangle((0, 0), self.classroom_width, self.classroom_height, 
                                 fill=False, color='black', linewidth=2))
        
        # Draw seats
        for seat_id, (x, y) in self.seat_positions.items():
            ax.plot(x, y, 'ko', markersize=8)
            ax.text(x, y + 20, seat_id, ha='center', va='bottom', fontsize=8)
        
        # Draw detected faces
        colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))
        for i, detection in enumerate(detections):
            if 'bbox' in detection:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Color based on attention
                attention_score = detection.get('attention', {}).get('attention_score', 0.5)
                color = plt.cm.RdYlGn(attention_score)  # Red (low attention) to Green (high attention)
                
                ax.plot(center_x, center_y, 'o', color=color, markersize=10)
                
                # Add identity label
                identity = detection.get('identity', 'Unknown')
                ax.text(center_x, center_y - 20, identity, ha='center', va='top', fontsize=8)
        
        ax.set_xlim(-50, self.classroom_width + 50)
        ax.set_ylim(-50, self.classroom_height + 50)
        ax.set_aspect('equal')
        ax.set_title('Classroom Layout with Face Detections')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        
        # Add colorbar for attention scores
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Score')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Classroom layout saved to: {save_path}")
        
        plt.show()
    
    def analyze_spatial_distribution(self, detections: List[Dict]) -> Dict:
        """Analyze spatial distribution of faces in classroom."""
        if not detections:
            return {}
        
        # Calculate basic statistics
        positions = []
        attention_scores = []
        emotions = []
        
        for detection in detections:
            if 'bbox' in detection:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                positions.append((center_x, center_y))
                
                attention_scores.append(detection.get('attention', {}).get('attention_score', 0))
                emotions.append(detection.get('dominant_emotion', 'neutral'))
        
        if not positions:
            return {}
        
        positions = np.array(positions)
        
        # Calculate spatial statistics
        mean_position = np.mean(positions, axis=0)
        std_position = np.std(positions, axis=0)
        
        # Calculate clustering
        from scipy.spatial.distance import pdist, squareform
        if len(positions) > 1:
            distances = pdist(positions)
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
        else:
            avg_distance = 0
            min_distance = 0
        
        # Calculate attention by region
        front_attention = []
        back_attention = []
        left_attention = []
        right_attention = []
        
        for i, (x, y) in enumerate(positions):
            attention = attention_scores[i]
            
            # Front/back regions (based on Y coordinate)
            if y < self.classroom_height / 2:
                front_attention.append(attention)
            else:
                back_attention.append(attention)
            
            # Left/right regions (based on X coordinate)
            if x < self.classroom_width / 2:
                left_attention.append(attention)
            else:
                right_attention.append(attention)
        
        return {
            'total_faces': len(positions),
            'mean_position': mean_position.tolist(),
            'std_position': std_position.tolist(),
            'average_distance': float(avg_distance),
            'minimum_distance': float(min_distance),
            'attention_by_region': {
                'front': np.mean(front_attention) if front_attention else 0,
                'back': np.mean(back_attention) if back_attention else 0,
                'left': np.mean(left_attention) if left_attention else 0,
                'right': np.mean(right_attention) if right_attention else 0
            },
            'emotion_distribution': self._count_emotions(emotions),
            'overall_attention': np.mean(attention_scores) if attention_scores else 0
        }
    
    def _count_emotions(self, emotions: List[str]) -> Dict[str, int]:
        """Count occurrences of each emotion."""
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        return emotion_counts
    
    def save_analysis_results(self, output_path: str) -> None:
        """Save analysis results to JSON file."""
        results = {
            'heatmaps': {k: v.tolist() for k, v in self.heatmaps.items()},
            'seat_assignments': self.seat_assignments,
            'classroom_dimensions': {
                'width': self.classroom_width,
                'height': self.classroom_height
            },
            'seat_positions': {k: list(v) for k, v in self.seat_positions.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to: {output_path}")
    
    def load_analysis_results(self, input_path: str) -> None:
        """Load analysis results from JSON file."""
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        self.heatmaps = {k: np.array(v) for k, v in results['heatmaps'].items()}
        self.seat_assignments = results['seat_assignments']
        
        logger.info(f"Analysis results loaded from: {input_path}")
    
    def generate_report(self, detections: List[Dict], output_dir: str) -> None:
        """Generate comprehensive layout analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate heatmaps
        heatmaps = self.generate_multiple_heatmaps(detections)
        
        # Assign seats
        seat_assignments = self.assign_seats(detections)
        
        # Analyze spatial distribution
        spatial_analysis = self.analyze_spatial_distribution(detections)
        
        # Save heatmap visualizations
        for heatmap_type, heatmap in heatmaps.items():
            save_path = output_path / f"heatmap_{heatmap_type}.png"
            self.visualize_heatmap(heatmap, f"{heatmap_type.title()} Heatmap", str(save_path))
        
        # Save classroom layout
        layout_path = output_path / "classroom_layout.png"
        self.visualize_classroom_layout(detections, str(layout_path))
        
        # Save analysis results
        results_path = output_path / "layout_analysis.json"
        self.save_analysis_results(str(results_path))
        
        # Generate summary report
        self._generate_summary_report(detections, spatial_analysis, output_path)
        
        logger.info(f"Layout analysis report generated in: {output_path}")
    
    def _generate_summary_report(self, detections: List[Dict], 
                               spatial_analysis: Dict, output_path: Path) -> None:
        """Generate a text summary report."""
        report_path = output_path / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("FaceClass Layout Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total faces detected: {spatial_analysis.get('total_faces', 0)}\n")
            f.write(f"Overall attention score: {spatial_analysis.get('overall_attention', 0):.3f}\n\n")
            
            f.write("Attention by Region:\n")
            attention_by_region = spatial_analysis.get('attention_by_region', {})
            for region, score in attention_by_region.items():
                f.write(f"  {region.capitalize()}: {score:.3f}\n")
            
            f.write("\nEmotion Distribution:\n")
            emotion_dist = spatial_analysis.get('emotion_distribution', {})
            for emotion, count in emotion_dist.items():
                f.write(f"  {emotion.capitalize()}: {count}\n")
            
            f.write(f"\nSpatial Statistics:\n")
            f.write(f"  Average distance between faces: {spatial_analysis.get('average_distance', 0):.1f} pixels\n")
            f.write(f"  Minimum distance between faces: {spatial_analysis.get('minimum_distance', 0):.1f} pixels\n")
            
            f.write(f"\nSeat Assignments:\n")
            for seat_id, assignment in self.seat_assignments.items():
                f.write(f"  {seat_id}: {assignment['identity']} (Attention: {assignment['attention_score']:.3f})\n")
        
        logger.info(f"Summary report saved to: {report_path}") 