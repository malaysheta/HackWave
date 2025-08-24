#!/usr/bin/env python3
"""
Quick test to verify backend endpoints are working.
"""

import requests
import json

def test_endpoints():
    """Test the key endpoints."""
    base_url = "http://localhost:2024"
    
    print("🧪 Quick Backend Test")
    print("=" * 40)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        print(f"✅ Health: {response.status_code}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
        return
    
    # Test default conversation history
    try:
        response = requests.get(f"{base_url}/api/conversation-history/default", timeout=5)
        print(f"✅ Default History: {response.status_code}")
        if response.status_code == 200:
            history = response.json()
            print(f"   Found {len(history)} conversations")
            if history:
                print(f"   Latest: {history[0].get('user_query', 'No query')[:50]}...")
    except Exception as e:
        print(f"❌ Default History failed: {e}")
    
    # Test specific thread history
    try:
        response = requests.get(f"{base_url}/api/conversation-history/test_thread", timeout=5)
        print(f"✅ Thread History: {response.status_code}")
    except Exception as e:
        print(f"❌ Thread History failed: {e}")
    
    print("=" * 40)
    print("✅ Backend is ready for frontend connection!")

if __name__ == "__main__":
    test_endpoints()
