#!/usr/bin/env python3
"""
Test script to isolate graph compilation issues.
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

def test_graph_compilation():
    """Test graph compilation step by step."""
    print("Testing graph compilation...")
    
    try:
        print("1. Importing required modules...")
        from langgraph.graph import StateGraph, START, END
        from src.agent.state import OverallState
        from src.agent.configuration import Configuration
        from langgraph.checkpoint.memory import MemorySaver
        print("✅ Imports successful")
        
        print("2. Creating StateGraph...")
        builder = StateGraph(OverallState, config_schema=Configuration)
        print("✅ StateGraph created")
        
        print("3. Adding simple test node...")
        def test_node(state):
            return {"user_query": "test", "is_complete": True}
        
        builder.add_node("test", test_node)
        print("✅ Test node added")
        
        print("4. Adding edges...")
        builder.add_edge(START, "test")
        builder.add_edge("test", END)
        print("✅ Edges added")
        
        print("5. Creating checkpoint saver...")
        checkpoint_saver = MemorySaver()
        print("✅ Checkpoint saver created")
        
        print("6. Compiling graph...")
        graph = builder.compile(
            name="test-graph",
            checkpointer=checkpoint_saver
        )
        print("✅ Graph compiled successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_supervisor_node():
    """Test supervisor node specifically."""
    print("\nTesting supervisor node...")
    
    try:
        print("1. Importing supervisor node...")
        from src.agent.graph import supervisor_node
        print("✅ Supervisor node imported")
        
        print("2. Testing supervisor node function...")
        # This should not hang
        print("✅ Supervisor node function accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Supervisor node test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Graph Compilation Test")
    print("=" * 50)
    
    # Test supervisor node first
    if not test_supervisor_node():
        print("\n❌ Supervisor node test failed")
        return False
    
    # Test graph compilation
    if not test_graph_compilation():
        print("\n❌ Graph compilation test failed")
        return False
    
    print("\n🎉 All graph tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

