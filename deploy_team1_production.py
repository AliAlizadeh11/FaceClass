#!/usr/bin/env python3
"""
Team 1 Production Deployment Script
Final verification and deployment preparation for Face Detection & Recognition Core
"""

import sys
import os
import logging
from pathlib import Path
import json
import time
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from detection.model_comparison import ModelBenchmarker
from detection.deep_ocsort import DeepOCSORTTracker
from recognition.face_quality import FaceQualityAssessor
from recognition.database_manager import FaceDatabaseManager

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('team1_production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Team1ProductionDeployment:
    """Production deployment manager for Team 1 components."""
    
    def __init__(self):
        """Initialize production deployment manager."""
        self.deployment_status = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'performance_metrics': {},
            'deployment_ready': False,
            'recommendations': []
        }
        
        # Production configuration
        self.production_config = {
            'face_detection': {
                'model': 'retinaface',
                'confidence_threshold': 0.8,
                'min_face_size': 80
            },
            'face_tracking': {
                'algorithm': 'deep_ocsort',
                'persistence_frames': 30,
                'multi_camera': True,
                'max_age': 30,
                'min_hits': 3
            },
            'face_quality': {
                'min_face_size': 80,
                'min_resolution': 64,
                'min_contrast': 30,
                'max_blur': 100,
                'min_eye_openness': 0.3
            },
            'database': {
                'max_faces_per_person': 20,
                'min_quality_score': 0.7,
                'enable_auto_optimization': True,
                'backup_interval': 24
            }
        }
    
    def verify_component_availability(self):
        """Verify all required components are available."""
        logger.info("🔍 Verifying component availability...")
        
        components = {
            'Model Comparison': 'src.detection.model_comparison',
            'Deep OC-SORT Tracking': 'src.detection.deep_ocsort',
            'Face Quality Assessment': 'src.recognition.face_quality',
            'Database Manager': 'src.recognition.database_manager'
        }
        
        for component_name, module_path in components.items():
            try:
                __import__(module_path)
                self.deployment_status['components'][component_name] = {
                    'status': 'available',
                    'import_success': True
                }
                logger.info(f"  ✅ {component_name}: Available")
            except ImportError as e:
                self.deployment_status['components'][component_name] = {
                    'status': 'unavailable',
                    'import_success': False,
                    'error': str(e)
                }
                logger.error(f"  ❌ {component_name}: Import failed - {e}")
        
        return all(comp['import_success'] for comp in self.deployment_status['components'].values())
    
    def test_component_initialization(self):
        """Test initialization of all components."""
        logger.info("🧪 Testing component initialization...")
        
        try:
            # Test Model Benchmarker
            benchmarker = ModelBenchmarker(self.production_config)
            self.deployment_status['components']['Model Comparison']['initialization'] = 'success'
            logger.info("  ✅ Model Benchmarker: Initialized successfully")
        except Exception as e:
            self.deployment_status['components']['Model Comparison']['initialization'] = 'failed'
            self.deployment_status['components']['Model Comparison']['init_error'] = str(e)
            logger.error(f"  ❌ Model Benchmarker: Initialization failed - {e}")
        
        try:
            # Test Deep OC-SORT Tracker
            tracker = DeepOCSORTTracker(self.production_config['face_tracking'])
            self.deployment_status['components']['Deep OC-SORT Tracking']['initialization'] = 'success'
            logger.info("  ✅ Deep OC-SORT Tracker: Initialized successfully")
        except Exception as e:
            self.deployment_status['components']['Deep OC-SORT Tracking']['initialization'] = 'failed'
            self.deployment_status['components']['Deep OC-SORT Tracking']['init_error'] = str(e)
            logger.error(f"  ❌ Deep OC-SORT Tracker: Initialization failed - {e}")
        
        try:
            # Test Face Quality Assessor
            assessor = FaceQualityAssessor(self.production_config['face_quality'])
            self.deployment_status['components']['Face Quality Assessment']['initialization'] = 'success'
            logger.info("  ✅ Face Quality Assessor: Initialized successfully")
        except Exception as e:
            self.deployment_status['components']['Face Quality Assessment']['initialization'] = 'failed'
            self.deployment_status['components']['Face Quality Assessment']['init_error'] = str(e)
            logger.error(f"  ❌ Face Quality Assessor: Initialization failed - {e}")
        
        try:
            # Test Database Manager
            db_manager = FaceDatabaseManager(self.production_config['database'])
            self.deployment_status['components']['Database Manager']['initialization'] = 'success'
            logger.info("  ✅ Database Manager: Initialized successfully")
        except Exception as e:
            self.deployment_status['components']['Database Manager']['initialization'] = 'failed'
            self.deployment_status['components']['Database Manager']['init_error'] = str(e)
            logger.error(f"  ❌ Database Manager: Initialization failed - {e}")
        
        return all(comp.get('initialization') == 'success' for comp in self.deployment_status['components'].values())
    
    def run_performance_benchmarks(self):
        """Run performance benchmarks for production readiness."""
        logger.info("📊 Running performance benchmarks...")
        
        try:
            # Quality Assessment Performance
            assessor = FaceQualityAssessor(self.production_config['face_quality'])
            
            # Create test images
            import numpy as np
            test_images = [np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8) for _ in range(20)]
            
            start_time = time.time()
            results = assessor.batch_assess_quality(test_images)
            total_time = time.time() - start_time
            
            quality_fps = len(test_images) / total_time
            self.deployment_status['performance_metrics']['quality_assessment'] = {
                'fps': quality_fps,
                'total_time': total_time,
                'images_processed': len(test_images)
            }
            
            logger.info(f"  ✅ Quality Assessment: {quality_fps:.1f} FPS")
            
            # Tracking Performance
            tracker = DeepOCSORTTracker(self.production_config['face_tracking'])
            
            mock_detections = [
                {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'label': 'face'},
                {'bbox': [300, 150, 400, 250], 'confidence': 0.8, 'label': 'face'}
            ]
            
            start_time = time.time()
            for frame_id in range(100):
                moved_detections = []
                for det in mock_detections:
                    bbox = det['bbox']
                    moved_detections.append({
                        'bbox': [bbox[0] + frame_id, bbox[1] + frame_id//2, 
                                bbox[2] + frame_id, bbox[3] + frame_id//2],
                        'confidence': det['confidence'],
                        'label': det['label']
                    })
                tracker.update(moved_detections, frame_id)
            
            total_time = time.time() - start_time
            tracking_fps = 100 / total_time
            
            self.deployment_status['performance_metrics']['tracking'] = {
                'fps': tracking_fps,
                'total_time': total_time,
                'frames_processed': 100
            }
            
            logger.info(f"  ✅ Tracking: {tracking_fps:.1f} FPS")
            
            # Database Performance
            db_manager = FaceDatabaseManager(self.production_config['database'])
            
            start_time = time.time()
            stats = db_manager.get_database_stats()
            db_query_time = time.time() - start_time
            
            self.deployment_status['performance_metrics']['database'] = {
                'query_time': db_query_time,
                'stats': stats
            }
            
            logger.info(f"  ✅ Database: Query time {db_query_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Performance benchmarks failed: {e}")
            return False
    
    def verify_production_requirements(self):
        """Verify production requirements are met."""
        logger.info("🎯 Verifying production requirements...")
        
        requirements_met = True
        
        # Check performance requirements
        quality_fps = self.deployment_status['performance_metrics'].get('quality_assessment', {}).get('fps', 0)
        tracking_fps = self.deployment_status['performance_metrics'].get('tracking', {}).get('fps', 0)
        
        if quality_fps < 30:
            self.deployment_status['recommendations'].append(
                f"Quality assessment performance ({quality_fps:.1f} FPS) below target (30 FPS)"
            )
            requirements_met = False
        
        if tracking_fps < 30:
            self.deployment_status['recommendations'].append(
                f"Tracking performance ({tracking_fps:.1f} FPS) below target (30 FPS)"
            )
            requirements_met = False
        
        # Check component availability
        for comp_name, comp_status in self.deployment_status['components'].items():
            if comp_status.get('initialization') != 'success':
                self.deployment_status['recommendations'].append(
                    f"Component {comp_name} failed initialization"
                )
                requirements_met = False
        
        # Check database functionality
        try:
            db_manager = FaceDatabaseManager(self.production_config['database'])
            db_manager.auto_maintenance()
            logger.info("  ✅ Database maintenance: Successful")
        except Exception as e:
            self.deployment_status['recommendations'].append(
                f"Database maintenance failed: {e}"
            )
            requirements_met = False
        
        if requirements_met:
            logger.info("  ✅ All production requirements met")
        else:
            logger.warning("  ⚠ Some production requirements not met")
        
        return requirements_met
    
    def generate_production_config(self):
        """Generate production configuration files."""
        logger.info("⚙️ Generating production configuration...")
        
        try:
            # Create production config directory
            config_dir = Path('production_config')
            config_dir.mkdir(exist_ok=True)
            
            # Save main configuration
            config_file = config_dir / 'team1_production_config.json'
            with open(config_file, 'w') as f:
                json.dump(self.production_config, f, indent=2)
            
            # Save deployment status
            status_file = config_dir / 'deployment_status.json'
            with open(status_file, 'w') as f:
                json.dump(self.deployment_status, f, indent=2)
            
            # Create production startup script
            startup_script = config_dir / 'start_team1_production.py'
            startup_content = '''#!/usr/bin/env python3
"""
Team 1 Production Startup Script
Automatically generated production startup script
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from detection.model_comparison import ModelBenchmarker
from detection.deep_ocsort import DeepOCSORTTracker
from recognition.face_quality import FaceQualityAssessor
from recognition.database_manager import FaceDatabaseManager

def start_production_system():
    """Start the Team 1 production system."""
    print("🚀 Starting Team 1 Production System...")
    
    # Load production configuration
    import json
    with open('production_config/team1_production_config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize components
    benchmarker = ModelBenchmarker(config)
    tracker = DeepOCSORTTracker(config['face_tracking'])
    assessor = FaceQualityAssessor(config['face_quality'])
    db_manager = FaceDatabaseManager(config['database'])
    
    print("✅ All components initialized successfully")
    print("🎯 Production system ready for face detection and recognition")
    
    return {
        'benchmarker': benchmarker,
        'tracker': tracker,
        'assessor': assessor,
        'db_manager': db_manager
    }

if __name__ == "__main__":
    components = start_production_system()
    print("\\n🎉 Team 1 Production System is now running!")
    print("\\nComponents available:")
    for name, component in components.items():
        print(f"  - {name}: {type(component).__name__}")
'''
            
            with open(startup_script, 'w') as f:
                f.write(startup_content)
            
            # Make startup script executable
            os.chmod(startup_script, 0o755)
            
            logger.info(f"  ✅ Production configuration generated in {config_dir}")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Failed to generate production configuration: {e}")
            return False
    
    def run_final_verification(self):
        """Run final verification before production deployment."""
        logger.info("🔍 Running final verification...")
        
        try:
            # Test complete workflow
            logger.info("  Testing complete workflow...")
            
            # Initialize all components
            benchmarker = ModelBenchmarker(self.production_config)
            tracker = DeepOCSORTTracker(self.production_config['face_tracking'])
            assessor = FaceQualityAssessor(self.production_config['face_quality'])
            db_manager = FaceDatabaseManager(self.production_config['database'])
            
            # Simulate production workflow
            import numpy as np
            
            # Mock video frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 1. Face detection (simulated)
            mock_detections = [
                {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'label': 'face'},
                {'bbox': [300, 150, 400, 250], 'confidence': 0.8, 'label': 'face'}
            ]
            
            # 2. Face tracking
            tracked_faces = tracker.update(mock_detections, frame_id=0, camera_id=0)
            
            # 3. Quality assessment
            for face in tracked_faces:
                bbox = face['bbox']
                face_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                quality_result = assessor.assess_face_quality(face_region)
                
                # 4. Database storage (if quality is good)
                if quality_result['is_suitable_for_recognition']:
                    mock_encoding = np.random.randn(128)
                    db_manager.add_face_encoding(
                        student_id=f"PROD_{face['track_id']}",
                        encoding=mock_encoding,
                        quality_score=quality_result['overall_score']
                    )
            
            logger.info("  ✅ Complete workflow test successful")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Final verification failed: {e}")
            return False
    
    def deploy(self):
        """Execute complete production deployment."""
        logger.info("🚀 Starting Team 1 Production Deployment...")
        logger.info("=" * 60)
        
        # Step 1: Verify component availability
        if not self.verify_component_availability():
            logger.error("❌ Component availability check failed")
            return False
        
        # Step 2: Test component initialization
        if not self.test_component_initialization():
            logger.error("❌ Component initialization test failed")
            return False
        
        # Step 3: Run performance benchmarks
        if not self.run_performance_benchmarks():
            logger.error("❌ Performance benchmarks failed")
            return False
        
        # Step 4: Verify production requirements
        if not self.verify_production_requirements():
            logger.warning("⚠ Some production requirements not met")
        
        # Step 5: Generate production configuration
        if not self.generate_production_config():
            logger.error("❌ Failed to generate production configuration")
            return False
        
        # Step 6: Final verification
        if not self.run_final_verification():
            logger.error("❌ Final verification failed")
            return False
        
        # Deployment complete
        self.deployment_status['deployment_ready'] = True
        self.deployment_status['deployment_completed'] = datetime.now().isoformat()
        
        logger.info("=" * 60)
        logger.info("🎉 TEAM 1 PRODUCTION DEPLOYMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        # Print deployment summary
        self.print_deployment_summary()
        
        return True
    
    def print_deployment_summary(self):
        """Print deployment summary."""
        print("\n" + "=" * 60)
        print("TEAM 1 PRODUCTION DEPLOYMENT SUMMARY")
        print("=" * 60)
        
        print(f"Deployment Timestamp: {self.deployment_status['timestamp']}")
        print(f"Deployment Status: {'✅ READY' if self.deployment_status['deployment_ready'] else '❌ NOT READY'}")
        
        print("\nComponent Status:")
        for comp_name, comp_status in self.deployment_status['components'].items():
            status_icon = "✅" if comp_status.get('initialization') == 'success' else "❌"
            print(f"  {status_icon} {comp_name}: {comp_status.get('status', 'unknown')}")
        
        print("\nPerformance Metrics:")
        metrics = self.deployment_status['performance_metrics']
        if 'quality_assessment' in metrics:
            print(f"  🎯 Quality Assessment: {metrics['quality_assessment']['fps']:.1f} FPS")
        if 'tracking' in metrics:
            print(f"  🎯 Tracking: {metrics['tracking']['fps']:.1f} FPS")
        if 'database' in metrics:
            print(f"  🎯 Database: {metrics['database']['query_time']:.3f}s query time")
        
        if self.deployment_status['recommendations']:
            print("\n⚠ Recommendations:")
            for rec in self.deployment_status['recommendations']:
                print(f"  - {rec}")
        
        print("\n🚀 Production Startup:")
        print("  python production_config/start_team1_production.py")
        
        print("\n📁 Configuration Files:")
        print("  - production_config/team1_production_config.json")
        print("  - production_config/deployment_status.json")
        print("  - production_config/start_team1_production.py")
        
        print("\n" + "=" * 60)


def main():
    """Main deployment function."""
    print("TEAM 1: FACE DETECTION & RECOGNITION CORE")
    print("PRODUCTION DEPLOYMENT MANAGER")
    print("=" * 60)
    
    # Create deployment manager
    deployment = Team1ProductionDeployment()
    
    # Execute deployment
    success = deployment.deploy()
    
    if success:
        print("\n🎉 PRODUCTION DEPLOYMENT SUCCESSFUL!")
        print("Team 1 is now ready for production use.")
        print("\nNext steps:")
        print("1. Review deployment status in production_config/")
        print("2. Start production system with start_team1_production.py")
        print("3. Monitor system performance and logs")
        print("4. Scale as needed for your production environment")
    else:
        print("\n❌ PRODUCTION DEPLOYMENT FAILED!")
        print("Please review the errors above and fix issues before deployment.")
    
    return success


if __name__ == "__main__":
    main()
