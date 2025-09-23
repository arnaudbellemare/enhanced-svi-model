#!/usr/bin/env python3
"""
Run Streamlit App for Enhanced SVI Model
Simple script to launch the Streamlit web application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    print("🚀 Launching Enhanced SVI Model - Streamlit App")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
    
    # Launch the app
    print("🌐 Starting web application...")
    print("📱 The app will open in your default browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("=" * 60)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main()
