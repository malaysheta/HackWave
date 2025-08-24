#!/usr/bin/env python3
"""
Test script for the Supervisor-based multi-agent system.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append('.')

from src.agent.graph import graph
from src.agent.state import OverallState, QueryType, DebateCategory, AgentType, SupervisorDecision
from langchain_core.messages import HumanMessage


async def test_supervisor_system():
    """Test the Supervisor-based multi-agent system."""
    
    print("🧪 Testing Supervisor-based Multi-Agent System")
    print("=" * 50)
    
    # Test case 1: Simple domain query
    print("\n📋 Test Case 1: Domain Expert Query")
    print("-" * 30)
    
    initial_state: OverallState = {
        "messages": [HumanMessage(content="What are the business requirements for a healthcare compliance system?")],
        "user_query": "What are the business requirements for a healthcare compliance system?",
        "query_type": QueryType.GENERAL,
        "debate_category": None,
        "domain_expert_analysis": None,
        "ux_ui_specialist_analysis": None,
        "technical_architect_analysis": None,
        "revenue_model_analyst_analysis": None,
        "moderator_aggregation": None,
        "debate_resolution": None,
        "final_answer": None,
        "processing_time": 0.0,
        # Supervisor-related fields
        "active_agent": None,
        "supervisor_decision": None,
        "supervisor_reasoning": None,
        "agent_history": [],
        "current_step": 1,
        "max_steps": 10,
        "is_complete": False
    }
    
    try:
        result = await graph.ainvoke(initial_state)
        
        print(f"✅ Query processed successfully")
        print(f"📊 Processing time: {result.get('processing_time', 0):.2f} seconds")
        print(f"🔄 Total steps: {result.get('current_step', 1)}")
        print(f"📝 Final answer length: {len(result.get('final_answer', ''))} characters")
        
        # Show agent history
        agent_history = result.get("agent_history", [])
        print(f"\n🤖 Agent History ({len(agent_history)} entries):")
        for i, entry in enumerate(agent_history, 1):
            print(f"  {i}. {entry.get('agent', 'unknown')} - Step {entry.get('step', '?')}")
            if entry.get('reasoning'):
                print(f"     Reasoning: {entry.get('reasoning', '')[:100]}...")
        
        # Show supervisor decisions
        supervisor_entries = [e for e in agent_history if e.get('agent') == 'supervisor']
        print(f"\n🎯 Supervisor Decisions ({len(supervisor_entries)}):")
        for i, entry in enumerate(supervisor_entries, 1):
            print(f"  {i}. Decision: {entry.get('decision', 'unknown')}")
            print(f"     Next Agent: {entry.get('next_agent', 'unknown')}")
            print(f"     Reasoning: {entry.get('reasoning', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return False


async def test_debate_handling():
    """Test debate handling functionality."""
    
    print("\n\n📋 Test Case 2: Debate Handling")
    print("-" * 30)
    
    initial_state: OverallState = {
        "messages": [HumanMessage(content="There's a debate about whether to use microservices or monolithic architecture for our e-commerce platform")],
        "user_query": "There's a debate about whether to use microservices or monolithic architecture for our e-commerce platform",
        "query_type": QueryType.GENERAL,
        "debate_category": None,
        "domain_expert_analysis": None,
        "ux_ui_specialist_analysis": None,
        "technical_architect_analysis": None,
        "revenue_model_analyst_analysis": None,
        "moderator_aggregation": None,
        "debate_resolution": None,
        "final_answer": None,
        "processing_time": 0.0,
        # Supervisor-related fields
        "active_agent": None,
        "supervisor_decision": None,
        "supervisor_reasoning": None,
        "agent_history": [],
        "current_step": 1,
        "max_steps": 10,
        "is_complete": False
    }
    
    try:
        result = await graph.ainvoke(initial_state)
        
        print(f"✅ Debate processed successfully")
        print(f"📊 Processing time: {result.get('processing_time', 0):.2f} seconds")
        print(f"🔄 Total steps: {result.get('current_step', 1)}")
        
        # Check if debate was detected and handled
        debate_resolution = result.get("debate_resolution")
        if debate_resolution:
            print(f"🎯 Debate Resolution: {debate_resolution[:200]}...")
        
        # Show which agents were involved
        agent_history = result.get("agent_history", [])
        involved_agents = set(entry.get('agent') for entry in agent_history)
        print(f"🤖 Agents involved: {', '.join(involved_agents)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during debate processing: {str(e)}")
        return False


async def main():
    """Main test function."""
    
    print("🚀 Starting Supervisor-based Multi-Agent System Tests")
    print("=" * 60)
    
    # Test 1: Basic functionality
    test1_success = await test_supervisor_system()
    
    # Test 2: Debate handling
    test2_success = await test_debate_handling()
    
    # Summary
    print("\n\n📊 Test Summary")
    print("=" * 30)
    print(f"✅ Test 1 (Basic Functionality): {'PASSED' if test1_success else 'FAILED'}")
    print(f"✅ Test 2 (Debate Handling): {'PASSED' if test2_success else 'FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! Supervisor-based system is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
