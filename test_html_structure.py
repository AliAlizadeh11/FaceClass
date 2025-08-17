#!/usr/bin/env python3
"""
Test script to verify HTML structure is correct
"""

def test_html_structure():
    """Test if the HTML template has the required elements."""
    
    print("Testing HTML structure...")
    
    try:
        # Read the HTML file
        with open('src/templates/video_frames_display.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for required elements
        required_elements = [
            'id="uploadArea"',
            'id="videoFile"',
            'id="processVideo"',
            'id="processingStatus"',
            'id="processingText"',
            'id="processingProgress"',
            'id="processingDetails"',
            'id="videoStats"',
            'id="totalFrames"',
            'id="processedFrames"',
            'id="videoFPS"',
            'id="totalFaces"',
            'id="framesGrid"'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in html_content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing elements: {missing_elements}")
            return False
        else:
            print("✅ All required HTML elements found")
        
        # Check for JavaScript class
        if 'class VideoFrameProcessor' in html_content:
            print("✅ VideoFrameProcessor class found")
        else:
            print("❌ VideoFrameProcessor class not found")
            return False
        
        # Check for required methods
        required_methods = [
            'initializeElements()',
            'bindEvents()',
            'handleFile(',
            'processVideo(',
            'displayResults(',
            'displayFrames(',
            'createFrameCard('
        ]
        
        missing_methods = []
        for method in required_methods:
            if method not in html_content:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print("✅ All required methods found")
        
        print("\n" + "="*50)
        print("✅ HTML STRUCTURE TEST PASSED!")
        print("✅ All required elements and methods are present")
        print("✅ The template should work correctly now")
        
        return True
        
    except FileNotFoundError:
        print("❌ HTML file not found")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_html_structure()
