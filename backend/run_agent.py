#!/usr/bin/env python3
"""
Example usage of the LangGraph agent with MongoDB memory persistence.

This script demonstrates:
1. Starting a session with a given thread_id
2. Running the graph with user queries
3. Demonstrating memory persistence between sessions
4. Showing how the agent recalls past conversations
"""

import asyncio
import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.graph import graph
from src.agent.state import OverallState
from src.agent.memory import create_memory_manager
from src.agent.configuration import Configuration

# Load environment variables
load_dotenv()

# Verify GEMINI_API_KEY is set
if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable is not set!")
    print("Please set it in your .env file or environment.")
    sys.exit(1)


async def run_agent_session(thread_id: str, user_query: str, max_steps: int = 10) -> Dict[str, Any]:
    """
    Run a single agent session with the given thread_id and user query.
    
    Args:
        thread_id: Unique identifier for the conversation thread
        user_query: The user's query to process
        max_steps: Maximum number of steps for the conversation
        
    Returns:
        Dictionary containing the final result
    """
    print(f"\n{'='*60}")
    print(f"Starting session with thread_id: {thread_id}")
    print(f"User Query: {user_query}")
    print(f"Max Steps: {max_steps}")
    print(f"{'='*60}")
    
    # Initialize memory manager
    memory_manager = create_memory_manager()
    
    # Get conversation history for context
    history = memory_manager.get_conversation_history(thread_id, limit=3)
    if history:
        print(f"\n📚 Found {len(history)} previous conversation(s) for this thread:")
        for i, entry in enumerate(reversed(history), 1):
            print(f"  {i}. Step {entry.get('current_step', 'N/A')}: {entry.get('user_query', 'No query')}")
    else:
        print("\n🆕 This is a new conversation thread")
    
    # Prepare initial state
    initial_state: OverallState = {
        "messages": [],
        "user_query": user_query,
        "query_type": None,
        "debate_category": None,
        "domain_expert_analysis": None,
        "ux_ui_specialist_analysis": None,
        "technical_architect_analysis": None,
        "revenue_model_analyst_analysis": None,
        "moderator_aggregation": None,
        "debate_resolution": None,
        "final_answer": None,
        "processing_time": 0.0,
        "active_agent": None,
        "supervisor_decision": None,
        "supervisor_reasoning": None,
        "agent_history": [],
        "current_step": 1,
        "max_steps": max_steps,
        "is_complete": False
    }
    
    # Prepare configuration with thread_id
    config = {
        "configurable": {
            "model": "gemini-2.0-flash",
            "thread_id": thread_id,
            "max_debate_resolution_time": 120,
            "enable_parallel_processing": True
        }
    }
    
    try:
        # Run the graph
        print(f"\n🚀 Running agent graph...")
        result = await graph.ainvoke(initial_state, config)
        
        # Save final state to memory
        memory_manager.save_conversation_memory(thread_id, result)
        
        print(f"\n✅ Session completed successfully!")
        print(f"Final Answer: {result.get('final_answer', 'No final answer generated')}")
        print(f"Total Steps: {result.get('current_step', 1)}")
        print(f"Processing Time: {result.get('processing_time', 0.0):.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during session: {e}")
        return {"error": str(e)}


async def demonstrate_memory_persistence():
    """
    Demonstrate memory persistence by running multiple sessions with the same thread_id.
    """
    print("🧠 LangGraph Agent with MongoDB Memory Persistence Demo")
    print("=" * 60)
    
    # Test thread ID
    test_thread_id = "demo_thread_001"
    
    # First session
    print("\n📝 Session 1: Initial query about a mobile app")
    result1 = await run_agent_session(
        thread_id=test_thread_id,
        user_query="I want to build a mobile app for food delivery. What are the key requirements?",
        max_steps=8
    )
    
    # Second session with the same thread_id
    print("\n📝 Session 2: Follow-up query about the same project")
    result2 = await run_agent_session(
        thread_id=test_thread_id,
        user_query="What about the technical architecture for this food delivery app?",
        max_steps=6
    )
    
    # Third session with the same thread_id
    print("\n📝 Session 3: Another follow-up about revenue model")
    result3 = await run_agent_session(
        thread_id=test_thread_id,
        user_query="How should we monetize this food delivery app?",
        max_steps=6
    )
    
    # Show memory summary
    print("\n📊 Memory Summary:")
    memory_manager = create_memory_manager()
    summary = memory_manager.get_thread_summary(test_thread_id)
    
    print(f"Thread ID: {summary['thread_id']}")
    print(f"Total Conversations: {summary['conversation_count']}")
    print(f"Last Updated: {summary['last_updated']}")
    
    return result1, result2, result3


async def demonstrate_new_thread():
    """
    Demonstrate a new thread to show isolation.
    """
    print("\n🆕 New Thread Demo")
    print("=" * 60)
    
    new_thread_id = "demo_thread_002"
    
    result = await run_agent_session(
        thread_id=new_thread_id,
        user_query="I want to create a social media platform for pet owners. What should I consider?",
        max_steps=8
    )
    
    return result


async def main():
    """
    Main function to run the demonstration.
    """
    try:
        # Demonstrate memory persistence
        await demonstrate_memory_persistence()
        
        # Demonstrate new thread
        await demonstrate_new_thread()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ MongoDB memory persistence")
        print("✅ Thread-based conversation isolation")
        print("✅ Conversation history retrieval")
        print("✅ Context-aware responses")
        print("✅ Checkpoint saving and loading")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())

