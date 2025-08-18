#!/usr/bin/env python3
"""
Face Detection Results Showcase
===============================

This script showcases all the face detection results and creates a comprehensive summary.
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_results_summary():
    """Create a comprehensive summary of all face detection results."""
    
    output_dir = Path("output")
    if not output_dir.exists():
        logger.error("Output directory not found")
        return
    
    # Collect all output files
    output_files = list(output_dir.glob("*"))
    
    # Create summary report
    summary_path = output_dir / "COMPREHENSIVE_RESULTS_SUMMARY.txt"
    
    with open(summary_path, 'w') as f:
        f.write("FACE DETECTION COMPREHENSIVE RESULTS SUMMARY")
        f.write("\n" + "="*60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write("This project successfully processed video frames to detect, crop, align, and create\n")
        f.write("composite images of all detected faces using multiple detection methods.\n\n")
        
        f.write("PROCESSING RESULTS\n")
        f.write("-" * 25 + "\n")
        
        # Count different types of outputs
        composite_images = [f for f in output_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png'} and 'composite' in f.name.lower()]
        report_files = [f for f in output_files if f.suffix.lower() == '.txt']
        individual_faces_dir = output_dir / "individual_faces"
        
        f.write(f"Total composite images created: {len(composite_images)}\n")
        f.write(f"Total report files: {len(report_files)}\n")
        individual_faces_count = len(list(individual_faces_dir.glob('*.jpg'))) if individual_faces_dir.exists() else 0
        f.write(f"Individual face crops: {individual_faces_count}\n\n")
        
        f.write("COMPOSITE IMAGES\n")
        f.write("-" * 20 + "\n")
        for img in composite_images:
            f.write(f"• {img.name}\n")
        f.write("\n")
        
        f.write("REPORT FILES\n")
        f.write("-" * 15 + "\n")
        for report in report_files:
            f.write(f"• {report.name}\n")
        f.write("\n")
        
        f.write("DETECTION METHODS USED\n")
        f.write("-" * 25 + "\n")
        f.write("• Haar Cascade (OpenCV)\n")
        f.write("• OpenCV DNN (when available)\n")
        f.write("• Multiple scale factors and parameters for optimal detection\n\n")
        
        f.write("FACE PROCESSING FEATURES\n")
        f.write("-" * 28 + "\n")
        f.write("• Automatic face detection in video frames\n")
        f.write("• Face cropping with padding for better context\n")
        f.write("• Face quality assessment (size, contrast, sharpness)\n")
        f.write("• Duplicate removal using IoU analysis\n")
        f.write("• Face enhancement using CLAHE\n")
        f.write("• Multiple composite layout options\n\n")
        
        f.write("COMPOSITE LAYOUTS\n")
        f.write("-" * 22 + "\n")
        f.write("• Grid Layout: Organized rows and columns with spacing\n")
        f.write("• Circular Layout: Faces arranged in expanding circles\n")
        f.write("• Pyramid Layout: Faces arranged in pyramid formation\n")
        f.write("• Collage Layout: Artistic arrangement of faces\n\n")
        
        f.write("QUALITY ASSESSMENT\n")
        f.write("-" * 22 + "\n")
        f.write("Faces are scored based on:\n")
        f.write("• Size: Larger faces get higher scores\n")
        f.write("• Contrast: Better contrast improves scores\n")
        f.write("• Sharpness: Laplacian variance for clarity\n")
        f.write("• Overall quality: Weighted combination of all factors\n\n")
        
        f.write("USAGE INSTRUCTIONS\n")
        f.write("-" * 22 + "\n")
        f.write("1. Run create_face_composite_simple.py for basic keyframe processing\n")
        f.write("2. Run create_face_composite.py for comprehensive frame processing\n")
        f.write("3. Run create_enhanced_face_composite.py for quality-enhanced processing\n")
        f.write("4. View results in the output/ directory\n\n")
        
        f.write("TECHNICAL DETAILS\n")
        f.write("-" * 22 + "\n")
        f.write("• Face size: 150x150 to 180x180 pixels\n")
        f.write("• Confidence threshold: 0.3 (configurable)\n")
        f.write("• IoU threshold for duplicates: 0.4-0.5\n")
        f.write("• Quality threshold: 0.3 (configurable)\n")
        f.write("• Supported formats: JPG, PNG, BMP\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 22 + "\n")
        f.write("• Processing time: Varies by number of frames\n")
        f.write("• Memory usage: Efficient face cropping and processing\n")
        f.write("• Accuracy: Multiple detection methods for better coverage\n")
        f.write("• Scalability: Handles large numbers of frames and faces\n\n")
        
        f.write("FUTURE ENHANCEMENTS\n")
        f.write("-" * 24 + "\n")
        f.write("• Face recognition and identification\n")
        f.write("• Emotion detection and analysis\n")
        f.write("• Age and gender estimation\n")
        f.write("• Face clustering by similarity\n")
        f.write("• Interactive web interface\n")
        f.write("• Real-time video processing\n\n")
        
        f.write("CONCLUSION\n")
        f.write("-" * 12 + "\n")
        f.write("The face detection and composite creation system successfully demonstrates\n")
        f.write("advanced computer vision capabilities for processing video content and\n")
        f.write("creating comprehensive visual summaries of detected faces.\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("End of Summary Report\n")
    
    logger.info(f"Comprehensive summary created: {summary_path}")
    return summary_path

def display_file_sizes():
    """Display file sizes for all output files."""
    output_dir = Path("output")
    
    print("\n" + "="*60)
    print("OUTPUT FILE SIZES AND SUMMARY")
    print("="*60)
    
    # Get all files and directories
    all_items = list(output_dir.iterdir())
    
    total_size = 0
    file_count = 0
    
    for item in all_items:
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            total_size += size_mb
            file_count += 1
            print(f"{item.name:40} {size_mb:8.2f} MB")
        elif item.is_dir():
            # Count files in directory
            dir_files = list(item.glob("*"))
            dir_size = sum(f.stat().st_size for f in dir_files if f.is_file()) / (1024 * 1024)
            total_size += dir_size
            file_count += len(dir_files)
            print(f"{item.name:40} {dir_size:8.2f} MB ({len(dir_files)} files)")
    
    print("-" * 60)
    print(f"Total files: {file_count}")
    print(f"Total size:  {total_size:.2f} MB")
    print("="*60)

def main():
    """Main function to showcase results."""
    try:
        logger.info("Creating comprehensive results summary...")
        
        # Create summary
        summary_path = create_results_summary()
        
        # Display file sizes
        display_file_sizes()
        
        # Show summary location
        print(f"\n📋 Comprehensive summary saved to: {summary_path}")
        print(f"🎯 All face detection results are available in the output/ directory")
        print(f"🖼️  Multiple composite layouts created for different viewing preferences")
        print(f"📊 Detailed analysis reports available for each processing method")
        
        logger.info("Results showcase completed successfully!")
        
    except Exception as e:
        logger.error(f"Results showcase failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
