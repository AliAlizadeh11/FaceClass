"""
Model Comparison and Benchmarking Script for Team 1
Compares performance of RetinaFace, YOLO, and MTCNN face detection models
"""

import cv2
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import psutil
import os

logger = logging.getLogger(__name__)


class ModelBenchmarker:
    """Benchmark different face detection models for performance comparison."""
    
    def __init__(self, config):
        """Initialize benchmarker with configuration."""
        self.config = config
        self.results = {}
        self.test_images = []
        self.test_videos = []
        
        # Performance metrics
        self.metrics = {
            'detection_accuracy': {},
            'processing_speed': {},
            'memory_usage': {},
            'model_size': {},
            'inference_time': {}
        }
    
    def load_test_data(self, data_path: str):
        """Load test images and videos for benchmarking."""
        data_path = Path(data_path)
        
        # Load test images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in image_extensions:
            self.test_images.extend(list(data_path.glob(f'**/*{ext}')))
        
        # Load test videos
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in video_extensions:
            self.test_videos.extend(list(data_path.glob(f'**/*{ext}')))
        
        logger.info(f"Loaded {len(self.test_images)} test images and {len(self.test_videos)} test videos")
    
    def benchmark_detection_models(self):
        """Run comprehensive benchmark of all detection models."""
        models = ['yolo', 'retinaface', 'mtcnn', 'opencv']
        
        for model in models:
            logger.info(f"Benchmarking {model.upper()} model...")
            try:
                self._benchmark_single_model(model)
            except Exception as e:
                logger.error(f"Failed to benchmark {model}: {e}")
                continue
        
        # Generate comparison report
        self._generate_comparison_report()
    
    def _benchmark_single_model(self, model_name: str):
        """Benchmark a single detection model."""
        from .face_tracker import FaceTracker
        
        # Create temporary config for this model
        temp_config = type('Config', (), {
            'get': lambda self, key, default=None: {
                'face_detection.model': model_name,
                'face_detection.confidence_threshold': 0.5,
                'face_detection.nms_threshold': 0.4,
                'face_detection.min_face_size': 20,
                'face_tracking.algorithm': 'simple_iou',
                'face_tracking.persistence_frames': 30,
                'face_tracking.multi_camera': False
            }.get(key, default),
            'get_path': lambda self, path_type: Path(__file__).parent.parent / 'models'
        })()
        
        # Initialize tracker
        tracker = FaceTracker(temp_config)
        
        # Benchmark on images
        image_results = self._benchmark_images(tracker, model_name)
        
        # Benchmark on videos
        video_results = self._benchmark_videos(tracker, model_name)
        
        # Store results
        self.results[model_name] = {
            'images': image_results,
            'videos': video_results,
            'overall': self._calculate_overall_metrics(image_results, video_results)
        }
    
    def _benchmark_images(self, tracker, model_name: str) -> Dict:
        """Benchmark model performance on test images."""
        results = {
            'total_images': len(self.test_images),
            'processed_images': 0,
            'total_detections': 0,
            'processing_times': [],
            'memory_usage': [],
            'detection_counts': []
        }
        
        for img_path in self.test_images[:10]:  # Limit to 10 images for speed
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Measure memory before
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Detect faces
                start_time = time.time()
                detections = tracker.detect_faces(image)
                processing_time = time.time() - start_time
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                # Store results
                results['processing_times'].append(processing_time)
                results['memory_usage'].append(memory_used)
                results['detection_counts'].append(len(detections))
                results['total_detections'] += len(detections)
                results['processed_images'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to process image {img_path}: {e}")
                continue
        
        # Calculate averages
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['avg_fps'] = 1.0 / results['avg_processing_time']
            results['avg_memory_usage'] = np.mean(results['memory_usage'])
            results['avg_detections_per_image'] = np.mean(results['detection_counts'])
        
        return results
    
    def _benchmark_videos(self, tracker, model_name: str) -> Dict:
        """Benchmark model performance on test videos."""
        results = {
            'total_videos': len(self.test_videos),
            'processed_videos': 0,
            'total_frames': 0,
            'total_detections': 0,
            'processing_times': [],
            'fps_metrics': [],
            'memory_usage': []
        }
        
        for video_path in self.test_videos[:3]:  # Limit to 3 videos for speed
            try:
                # Process video
                start_time = time.time()
                detections = tracker.process_video(str(video_path))
                processing_time = time.time() - start_time
                
                # Get video info
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                # Calculate metrics
                processing_fps = frame_count / processing_time if processing_time > 0 else 0
                
                # Store results
                results['processing_times'].append(processing_time)
                results['fps_metrics'].append(processing_fps)
                results['total_frames'] += frame_count
                results['total_detections'] += len(detections)
                results['processed_videos'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to process video {video_path}: {e}")
                continue
        
        # Calculate averages
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['avg_fps'] = np.mean(results['fps_metrics'])
        
        return results
    
    def _calculate_overall_metrics(self, image_results: Dict, video_results: Dict) -> Dict:
        """Calculate overall performance metrics."""
        overall = {}
        
        # Image metrics
        if image_results['processed_images'] > 0:
            overall['image_processing_fps'] = image_results.get('avg_fps', 0)
            overall['avg_detections_per_image'] = image_results.get('avg_detections_per_image', 0)
            overall['image_memory_usage_mb'] = image_results.get('avg_memory_usage', 0)
        
        # Video metrics
        if video_results['processed_videos'] > 0:
            overall['video_processing_fps'] = video_results.get('avg_fps', 0)
            overall['total_frames_processed'] = video_results['total_frames']
            overall['total_detections'] = video_results['total_detections']
        
        # Combined metrics
        overall['total_processing_time'] = sum(image_results.get('processing_times', [])) + sum(video_results.get('processing_times', []))
        overall['total_images_processed'] = image_results['processed_images']
        overall['total_videos_processed'] = video_results['processed_videos']
        
        return overall
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        report = {
            'benchmark_summary': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_models_tested': len(self.results),
                'test_data_summary': {
                    'total_images': len(self.test_images),
                    'total_videos': len(self.test_videos)
                }
            },
            'model_performance': {},
            'recommendations': {}
        }
        
        # Analyze each model's performance
        for model_name, results in self.results.items():
            report['model_performance'][model_name] = {
                'overall_score': self._calculate_model_score(results),
                'strengths': self._identify_strengths(results),
                'weaknesses': self._identify_weaknesses(results),
                'best_use_case': self._determine_best_use_case(results)
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        # Save report
        report_path = Path(__file__).parent.parent / 'output' / 'model_comparison_report.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to: {report_path}")
        
        # Print summary
        self._print_summary(report)
    
    def _calculate_model_score(self, results: Dict) -> float:
        """Calculate overall score for a model (0-100)."""
        score = 0
        
        # Image processing score (40%)
        if 'images' in results and results['images']['processed_images'] > 0:
            img_fps = results['images'].get('avg_fps', 0)
            score += min(img_fps / 30.0, 1.0) * 40  # Normalize to 30 FPS target
        
        # Video processing score (40%)
        if 'videos' in results and results['videos']['processed_videos'] > 0:
            vid_fps = results['videos'].get('avg_fps', 0)
            score += min(vid_fps / 30.0, 1.0) * 40  # Normalize to 30 FPS target
        
        # Detection quality score (20%)
        if 'images' in results:
            avg_detections = results['images'].get('avg_detections_per_image', 0)
            score += min(avg_detections / 5.0, 1.0) * 20  # Normalize to 5 detections per image
        
        return round(score, 2)
    
    def _identify_strengths(self, results: Dict) -> List[str]:
        """Identify model strengths."""
        strengths = []
        
        if 'images' in results:
            if results['images'].get('avg_fps', 0) > 25:
                strengths.append("High image processing speed")
            if results['images'].get('avg_detections_per_image', 0) > 3:
                strengths.append("Good detection coverage")
        
        if 'videos' in results:
            if results['videos'].get('avg_fps', 0) > 25:
                strengths.append("High video processing speed")
        
        if not strengths:
            strengths.append("Stable performance")
        
        return strengths
    
    def _identify_weaknesses(self, results: Dict) -> List[str]:
        """Identify model weaknesses."""
        weaknesses = []
        
        if 'images' in results:
            if results['images'].get('avg_fps', 0) < 15:
                weaknesses.append("Slow image processing")
            if results['images'].get('avg_detections_per_image', 0) < 1:
                weaknesses.append("Low detection rate")
        
        if 'videos' in results:
            if results['videos'].get('avg_fps', 0) < 15:
                weaknesses.append("Slow video processing")
        
        if not weaknesses:
            weaknesses.append("No significant weaknesses identified")
        
        return weaknesses
    
    def _determine_best_use_case(self, results: Dict) -> str:
        """Determine the best use case for the model."""
        if 'images' in results and 'videos' in results:
            img_fps = results['images'].get('avg_fps', 0)
            vid_fps = results['videos'].get('avg_fps', 0)
            
            if img_fps > vid_fps * 1.5:
                return "Image processing focused"
            elif vid_fps > img_fps * 1.5:
                return "Video processing focused"
            else:
                return "Balanced performance"
        
        return "General purpose"
    
    def _generate_recommendations(self) -> Dict:
        """Generate recommendations based on benchmark results."""
        recommendations = {
            'best_overall': None,
            'best_speed': None,
            'best_accuracy': None,
            'best_memory_efficient': None,
            'deployment_suggestions': []
        }
        
        # Find best performers
        scores = {name: results['overall']['overall_score'] for name, results in self.results.items()}
        if scores:
            recommendations['best_overall'] = max(scores, key=scores.get)
        
        # Find fastest model
        speeds = {}
        for name, results in self.results.items():
            if 'images' in results:
                speeds[name] = results['images'].get('avg_fps', 0)
        if speeds:
            recommendations['best_speed'] = max(speeds, key=speeds.get)
        
        # Find most accurate model (based on detection count)
        accuracies = {}
        for name, results in self.results.items():
            if 'images' in results:
                accuracies[name] = results['images'].get('avg_detections_per_image', 0)
        if accuracies:
            recommendations['best_accuracy'] = max(accuracies, key=accuracies.get)
        
        # Find most memory efficient
        memory_usage = {}
        for name, results in self.results.items():
            if 'images' in results:
                memory_usage[name] = results['images'].get('avg_memory_usage', float('inf'))
        if memory_usage:
            recommendations['best_memory_efficient'] = min(memory_usage, key=memory_usage.get)
        
        # Generate deployment suggestions
        for name, results in self.results.items():
            score = results['overall']['overall_score']
            if score >= 80:
                recommendations['deployment_suggestions'].append(f"{name.upper()}: Production ready")
            elif score >= 60:
                recommendations['deployment_suggestions'].append(f"{name.upper()}: Development/testing use")
            else:
                recommendations['deployment_suggestions'].append(f"{name.upper()}: Needs optimization")
        
        return recommendations
    
    def _print_summary(self, report: Dict):
        """Print benchmark summary to console."""
        print("\n" + "="*60)
        print("FACE DETECTION MODEL BENCHMARK RESULTS")
        print("="*60)
        
        print(f"\nBenchmark completed: {report['benchmark_summary']['timestamp']}")
        print(f"Models tested: {report['benchmark_summary']['total_models_tested']}")
        
        print("\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        for model_name, performance in report['model_performance'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Overall Score: {performance['overall_score']}/100")
            print(f"  Strengths: {', '.join(performance['strengths'])}")
            print(f"  Weaknesses: {', '.join(performance['weaknesses'])}")
            print(f"  Best Use Case: {performance['best_use_case']}")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 20)
        print(f"Best Overall: {report['recommendations']['best_overall']}")
        print(f"Best Speed: {report['recommendations']['best_speed']}")
        print(f"Best Accuracy: {report['recommendations']['best_accuracy']}")
        print(f"Most Memory Efficient: {report['recommendations']['best_memory_efficient']}")
        
        print("\nDeployment Suggestions:")
        for suggestion in report['recommendations']['deployment_suggestions']:
            print(f"  • {suggestion}")
        
        print("\n" + "="*60)


def run_benchmark(config, data_path: str = None):
    """Run the complete benchmark suite."""
    if data_path is None:
        data_path = Path(__file__).parent.parent / 'data'
    
    benchmarker = ModelBenchmarker(config)
    benchmarker.load_test_data(str(data_path))
    benchmarker.benchmark_detection_models()
    
    return benchmarker.results


if __name__ == "__main__":
    # Example usage
    config = type('Config', (), {
        'get': lambda self, key, default=None: default,
        'get_path': lambda self, path_type: Path(__file__).parent.parent / 'models'
    })()
    
    results = run_benchmark(config)
    print("Benchmark completed!")
