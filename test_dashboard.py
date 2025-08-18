#!/usr/bin/env python3
"""
Test script to verify dashboard functionality and handle 304 errors.
"""

import sys
import os
import time
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_dashboard_health():
    """Test dashboard health and functionality."""
    try:
        from dashboard.dashboard_ui import DashboardUI
        
        # Create a mock config
        class MockConfig:
            def get(self, key, default=None):
                return default
        
        config = MockConfig()
        
        # Initialize dashboard
        dashboard = DashboardUI(config)
        
        print("✅ DashboardUI initialized successfully")
        
        # Test health check endpoint
        try:
            # Start dashboard in background
            dashboard.start_background()
            time.sleep(2)  # Wait for dashboard to start
            
            # Test health endpoint
            health_url = f"http://{dashboard.host}:{dashboard.port}/health"
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                print("✅ Dashboard health check passed")
                print(f"   Health response: {response.json()}")
            else:
                print(f"❌ Dashboard health check failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not connect to dashboard: {e}")
            print("   This is normal if the dashboard is not running")
        
        # Test video upload functionality
        print("\n🔍 Testing video upload functionality...")
        
        # Test the save_uploaded_video method
        mock_contents = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAB1tZGF0AAACmwYF//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE0OCByMjYzOSBhOWE1OTRlIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAxNSAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcWw9MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTYgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEwIHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAAAGWWEhAA3//728P4FNjuZQQAAAu8AAAUcAElkAAAbAAAGUAAAAAA="
        mock_filename = "test_video.mp4"
        
        result = dashboard._save_uploaded_video(mock_contents, mock_filename)
        
        if result:
            print(f"✅ Video upload test successful: {result}")
            # Clean up test file
            if os.path.exists(result):
                os.remove(result)
                print("✅ Test file cleaned up")
        else:
            print("❌ Video upload test failed")
            
        print("\n🎯 Dashboard test completed!")
        print("📝 To start the dashboard, run: python3 src/main.py --mode dashboard")
        print("🌐 Then open: http://localhost:8080")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dashboard_health() 