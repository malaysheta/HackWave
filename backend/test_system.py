#!/usr/bin/env python3
"""
Test script for the Multi-Agent Product Requirements Refinement System.

This script tests the basic functionality of the multi-agent system.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import graph
from agent.state import OverallState, QueryType
from langchain_core.messages import HumanMessage


def test_basic_functionality():
    """Test the basic functionality of the multi-agent system."""
    
    print("🧪 Testing Multi-Agent Product Requirements Refinement System")
    print("=" * 60)
    
    # Test query
    test_query = "What are the requirements for a mobile banking application?"
    
    print(f"Test Query: {test_query}")
    print("-" * 60)
    
    # Prepare the initial state
    initial_state: OverallState = {
        "messages": [HumanMessage(content=test_query)],
        "user_query": test_query,
        "query_type": QueryType.GENERAL,
        "debate_category": None,
        "domain_expert_analysis": None,
        "ux_ui_specialist_analysis": None,
        "technical_architect_analysis": None,
        "moderator_aggregation": None,
        "debate_resolution": None,
        "final_answer": None,
        "processing_time": 0.0
    }
    
    try:
        print("🔄 Running multi-agent analysis...")
        
        # Run the graph
        result = graph.invoke(initial_state)
        
        print("✅ Analysis completed successfully!")
        print()
        
        # Check if we got the expected outputs
        print("📊 Results Summary:")
        print(f"  - Query Type: {result.get('query_type', 'Unknown')}")
        print(f"  - Domain Analysis: {'✅' if result.get('domain_expert_analysis') else '❌'}")
        print(f"  - UX/UI Analysis: {'✅' if result.get('ux_ui_specialist_analysis') else '❌'}")
        print(f"  - Technical Analysis: {'✅' if result.get('technical_architect_analysis') else '❌'}")
        print(f"  - Moderator Aggregation: {'✅' if result.get('moderator_aggregation') else '❌'}")
        print(f"  - Final Answer: {'✅' if result.get('final_answer') else '❌'}")
        
        # Display final answer
        if result.get("final_answer"):
            print("\n📝 Final Answer Preview:")
            print("-" * 30)
            answer = result["final_answer"]
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(preview)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False


def test_debate_handling():
    """Test the debate handling functionality."""
    
    print("\n🧪 Testing Debate Handling")
    print("=" * 60)
    
    # Test debate query
    test_debate = "There's a debate about whether to use React Native or Flutter for our mobile app. The team is divided on performance vs development speed."
    
    print(f"Test Debate: {test_debate}")
    print("-" * 60)
    
    # Prepare the initial state
    initial_state: OverallState = {
        "messages": [HumanMessage(content=test_debate)],
        "user_query": test_debate,
        "query_type": QueryType.GENERAL,
        "debate_category": None,
        "domain_expert_analysis": None,
        "ux_ui_specialist_analysis": None,
        "technical_architect_analysis": None,
        "moderator_aggregation": None,
        "debate_resolution": None,
        "final_answer": None,
        "processing_time": 0.0
    }
    
    try:
        print("🔄 Running debate analysis...")
        
        # Run the graph
        result = graph.invoke(initial_state)
        
        print("✅ Debate analysis completed successfully!")
        print()
        
        # Check debate-specific outputs
        print("📊 Debate Results Summary:")
        print(f"  - Debate Category: {result.get('debate_category', 'None')}")
        print(f"  - Debate Resolution: {'✅' if result.get('debate_resolution') else '❌'}")
        print(f"  - Final Answer: {'✅' if result.get('final_answer') else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during debate testing: {str(e)}")
        return False


def main():
    """Main test function."""
    
    # Check if GEMINI_API_KEY is set
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY environment variable is not set!")
        print("Please set your Gemini API key before running tests.")
        return False
    
    print("🚀 Starting Multi-Agent System Tests")
    print("=" * 60)
    
    # Run basic functionality test
    basic_test_passed = test_basic_functionality()
    
    # Run debate handling test
    debate_test_passed = test_debate_handling()
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print(f"  - Basic Functionality: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
    print(f"  - Debate Handling: {'✅ PASSED' if debate_test_passed else '❌ FAILED'}")
    
    if basic_test_passed and debate_test_passed:
        print("\n🎉 All tests passed! The multi-agent system is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
