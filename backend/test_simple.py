#!/usr/bin/env python3
"""
Simple test script to isolate import issues.
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

def test_imports():
    """Test imports step by step."""
    print("Testing imports step by step...")
    
    try:
        print("1. Testing memory import...")
        from src.agent.memory import create_memory_manager
        print("✅ Memory import successful")
    except Exception as e:
        print(f"❌ Memory import failed: {e}")
        return False
    
    try:
        print("2. Testing state import...")
        from src.agent.state import OverallState
        print("✅ State import successful")
    except Exception as e:
        print(f"❌ State import failed: {e}")
        return False
    
    try:
        print("3. Testing configuration import...")
        from src.agent.configuration import Configuration
        print("✅ Configuration import successful")
    except Exception as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    
    try:
        print("4. Testing prompts import...")
        from src.agent.prompts import get_current_date
        print("✅ Prompts import successful")
    except Exception as e:
        print(f"❌ Prompts import failed: {e}")
        return False
    
    try:
        print("5. Testing tools_and_schemas import...")
        from src.agent.tools_and_schemas import QueryClassification
        print("✅ Tools and schemas import successful")
    except Exception as e:
        print(f"❌ Tools and schemas import failed: {e}")
        return False
    
    return True

def test_memory_system():
    """Test memory system without graph."""
    print("\nTesting memory system...")
    
    try:
        from src.agent.memory import create_memory_manager
        
        # Create memory manager
        memory_manager = create_memory_manager()
        print("✅ Memory manager created")
        
        # Test basic operations
        test_thread_id = "test_simple_001"
        test_state = {
            "user_query": "Test query",
            "current_step": 1,
            "agent_history": [],
            "is_complete": False
        }
        
        # Test save
        success = memory_manager.save_conversation_memory(test_thread_id, test_state)
        if success:
            print("✅ Memory save successful")
        else:
            print("❌ Memory save failed")
            return False
        
        # Test retrieve
        history = memory_manager.get_conversation_history(test_thread_id, limit=5)
        if history:
            print("✅ Memory retrieve successful")
        else:
            print("❌ Memory retrieve failed")
            return False
        
        # Cleanup
        memory_manager.clear_thread_memory(test_thread_id)
        memory_manager.close()
        print("✅ Memory cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Simple Import and Memory Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return False
    
    # Test memory system
    if not test_memory_system():
        print("\n❌ Memory system test failed")
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

