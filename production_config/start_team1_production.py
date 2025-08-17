#!/usr/bin/env python3
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
    print("\n🎉 Team 1 Production System is now running!")
    print("\nComponents available:")
    for name, component in components.items():
        print(f"  - {name}: {type(component).__name__}")
