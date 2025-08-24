#!/usr/bin/env python3
"""
Simple test script to verify backend API connection and conversation history.
"""

import requests
import json
import time

def test_backend_connection():
    """Test if the backend is responding."""
    try:
        # Test health endpoint
        response = requests.get("http://localhost:2024/api/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        
        # Test conversation history endpoint
        test_thread_id = "test_thread_123"
        response = requests.get(f"http://localhost:2024/api/conversation-history/{test_thread_id}", timeout=5)
        print(f"✅ Conversation history: {response.status_code}")
        if response.status_code == 200:
            history = response.json()
            print(f"History entries: {len(history)}")
            for entry in history[:2]:  # Show first 2 entries
                print(f"  - {entry.get('user_query', 'No query')} ({entry.get('timestamp', 'No timestamp')})")
        
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Backend not responding on port 2024")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_streaming_endpoint():
    """Test the streaming endpoint with thread_id."""
    try:
        test_data = {
            "query": "I want to build a mobile app for food delivery",
            "query_type": "general",
            "thread_id": "test_thread_456"
        }
        
        print("🔄 Testing streaming endpoint...")
        response = requests.post(
            "http://localhost:2024/api/refine-requirements/stream",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30,
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Streaming endpoint working")
            # Read a few lines to verify streaming
            for i, line in enumerate(response.iter_lines()):
                if i >= 5:  # Just read first 5 lines
                    break
                if line:
                    print(f"  Stream: {line.decode()}")
        else:
            print(f"❌ Streaming endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Streaming test error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Backend API Connection")
    print("=" * 50)
    
    if test_backend_connection():
        test_streaming_endpoint()
    else:
        print("Backend not available. Make sure to run 'cd backend && langgraph dev --allow-blocking'")
