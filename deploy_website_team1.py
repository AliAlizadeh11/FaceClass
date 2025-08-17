#!/usr/bin/env python3
"""
Team 1 Website Deployment Script
Deploys the FaceClass website with all Team 1 components integrated
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask',
        'opencv-python',
        'numpy',
        'pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_team1_components():
    """Check if Team 1 components are available."""
    print("\n🔍 Checking Team 1 components...")
    
    team1_modules = [
        'src.detection.model_comparison',
        'src.detection.deep_ocsort',
        'src.recognition.face_quality',
        'src.recognition.database_manager'
    ]
    
    missing_modules = []
    for module in team1_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            missing_modules.append(module)
            print(f"  ❌ {module} - {e}")
    
    if missing_modules:
        print(f"\n⚠ Missing Team 1 modules: {len(missing_modules)}")
        print("Some features may not be available")
        return False
    
    return True

def start_website():
    """Start the Flask website."""
    print("\n🚀 Starting FaceClass website with Team 1 features...")
    
    # Change to src directory
    os.chdir('src')
    
    # Start Flask app
    try:
        print("  Starting Flask application...")
        print("  Website will be available at: http://localhost:5000")
        print("  Team 1 Dashboard: http://localhost:5000/team1-dashboard")
        print("  Press Ctrl+C to stop the server")
        print("\n" + "="*60)
        
        # Start Flask app
        subprocess.run([sys.executable, 'app.py'])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Website stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting website: {e}")
        return False
    
    return True

def main():
    """Main deployment function."""
    print("="*60)
    print("TEAM 1 WEBSITE DEPLOYMENT")
    print("FaceClass with Enhanced Face Detection & Recognition")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return False
    
    # Check Team 1 components
    team1_available = check_team1_components()
    
    if team1_available:
        print("\n🎉 All Team 1 components are available!")
    else:
        print("\n⚠ Some Team 1 components are missing. Website will run with limited functionality.")
    
    # Start website
    if start_website():
        print("\n✅ Website deployment completed successfully!")
        return True
    else:
        print("\n❌ Website deployment failed!")
        return False

if __name__ == "__main__":
    main()
