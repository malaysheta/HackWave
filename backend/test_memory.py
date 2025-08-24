#!/usr/bin/env python3
"""
Simple test script to verify MongoDB memory system functionality.
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.memory import create_memory_manager

# Load environment variables
load_dotenv()


def test_memory_manager():
    """Test the MongoDB memory manager functionality."""
    print("🧪 Testing MongoDB Memory Manager")
    print("=" * 50)
    
    try:
        # Create memory manager
        print("1. Creating memory manager...")
        memory_manager = create_memory_manager()
        print("✅ Memory manager created successfully")
        
        # Test thread ID
        test_thread_id = "test_thread_001"
        
        # Test saving conversation memory
        print("\n2. Testing conversation memory save...")
        test_state = {
            "user_query": "Test query for memory system",
            "current_step": 1,
            "agent_history": [{"step": 1, "agent": "test", "timestamp": 1234567890}],
            "active_agent": "test_agent",
            "is_complete": False,
            "processing_time": 1.5
        }
        
        success = memory_manager.save_conversation_memory(test_thread_id, test_state)
        if success:
            print("✅ Conversation memory saved successfully")
        else:
            print("❌ Failed to save conversation memory")
            return False
        
        # Test retrieving conversation history
        print("\n3. Testing conversation history retrieval...")
        history = memory_manager.get_conversation_history(test_thread_id, limit=5)
        if history:
            print(f"✅ Retrieved {len(history)} conversation history entries")
            for entry in history:
                print(f"   - Query: {entry.get('user_query', 'N/A')}")
                print(f"   - Step: {entry.get('current_step', 'N/A')}")
        else:
            print("❌ Failed to retrieve conversation history")
            return False
        
        # Test saving memory context
        print("\n4. Testing memory context save...")
        test_context = {
            "project_name": "Test Project",
            "user_preferences": ["feature1", "feature2"],
            "notes": "This is a test context"
        }
        
        success = memory_manager.save_memory_context(test_thread_id, test_context)
        if success:
            print("✅ Memory context saved successfully")
        else:
            print("❌ Failed to save memory context")
            return False
        
        # Test retrieving memory context
        print("\n5. Testing memory context retrieval...")
        context = memory_manager.get_memory_context(test_thread_id)
        if context:
            print("✅ Memory context retrieved successfully")
            print(f"   - Project: {context.get('project_name', 'N/A')}")
            print(f"   - Preferences: {context.get('user_preferences', [])}")
        else:
            print("❌ Failed to retrieve memory context")
            return False
        
        # Test thread summary
        print("\n6. Testing thread summary...")
        summary = memory_manager.get_thread_summary(test_thread_id)
        if summary:
            print("✅ Thread summary retrieved successfully")
            print(f"   - Thread ID: {summary.get('thread_id', 'N/A')}")
            print(f"   - Conversation Count: {summary.get('conversation_count', 0)}")
            print(f"   - Last Updated: {summary.get('last_updated', 'N/A')}")
        else:
            print("❌ Failed to retrieve thread summary")
            return False
        
        # Test memory isolation with different thread
        print("\n7. Testing memory isolation...")
        different_thread_id = "test_thread_002"
        different_history = memory_manager.get_conversation_history(different_thread_id, limit=5)
        if not different_history:
            print("✅ Memory isolation working correctly (no history for different thread)")
        else:
            print("❌ Memory isolation failed (found history for different thread)")
            return False
        
        print("\n🎉 All memory manager tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Memory manager test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        try:
            memory_manager.clear_thread_memory(test_thread_id)
            memory_manager.clear_thread_memory(different_thread_id)
            memory_manager.close()
            print("\n🧹 Cleanup completed")
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")


def test_mongodb_connection():
    """Test MongoDB connection."""
    print("\n🔌 Testing MongoDB Connection")
    print("=" * 50)
    
    try:
        from pymongo import MongoClient
        
        # Test connection
        client = MongoClient("mongodb://localhost:27017/Hackwave")
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        
        # Test database access
        db = client.get_database()
        collections = db.list_collection_names()
        print(f"✅ Database access successful. Collections: {collections}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n💡 Make sure MongoDB is running on localhost:27017")
        print("   You can start MongoDB with: mongod")
        return False


def main():
    """Main test function."""
    print("🧪 LangGraph MongoDB Memory System Test")
    print("=" * 60)
    
    # Test MongoDB connection first
    if not test_mongodb_connection():
        print("\n❌ Cannot proceed without MongoDB connection")
        return False
    
    # Test memory manager
    if test_memory_manager():
        print("\n🎉 All tests passed! Memory system is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

