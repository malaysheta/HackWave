#!/usr/bin/env python3
"""
Debug script to identify issues with the multi-agent system.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all imports to identify issues."""
    print("🔍 Testing imports...")
    
    try:
        print("  - Testing agent imports...")
        from agent.state import OverallState, QueryType, DebateCategory
        print("    ✅ agent.state imported successfully")
    except Exception as e:
        print(f"    ❌ agent.state import failed: {e}")
        return False
    
    try:
        print("  - Testing agent.tools_and_schemas...")
        from agent.tools_and_schemas import (
            QueryClassification,
            DomainExpertAnalysis,
            UXUISpecialistAnalysis,
            TechnicalArchitectAnalysis,
            ModeratorAggregation,
            DebateAnalysis,
        )
        print("    ✅ agent.tools_and_schemas imported successfully")
    except Exception as e:
        print(f"    ❌ agent.tools_and_schemas import failed: {e}")
        return False
    
    try:
        print("  - Testing agent.configuration...")
        from agent.configuration import Configuration
        print("    ✅ agent.configuration imported successfully")
    except Exception as e:
        print(f"    ❌ agent.configuration import failed: {e}")
        return False
    
    try:
        print("  - Testing agent.prompts...")
        from agent.prompts import (
            get_current_date,
            query_classification_instructions,
            domain_expert_instructions,
            ux_ui_specialist_instructions,
            technical_architect_instructions,
            moderator_aggregation_instructions,
            debate_analysis_instructions,
            final_answer_instructions,
        )
        print("    ✅ agent.prompts imported successfully")
    except Exception as e:
        print(f"    ❌ agent.prompts import failed: {e}")
        return False
    
    try:
        print("  - Testing agent.utils...")
        from agent.utils import get_user_query, format_agent_response
        print("    ✅ agent.utils imported successfully")
    except Exception as e:
        print(f"    ❌ agent.utils import failed: {e}")
        return False
    
    try:
        print("  - Testing agent.graph...")
        from agent.graph import graph
        print("    ✅ agent.graph imported successfully")
    except Exception as e:
        print(f"    ❌ agent.graph import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables."""
    print("\n🔍 Testing environment...")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"  ✅ GEMINI_API_KEY is set (length: {len(gemini_key)})")
    else:
        print("  ❌ GEMINI_API_KEY is not set")
        return False
    
    return True

def test_graph_compilation():
    """Test if the graph compiles successfully."""
    print("\n🔍 Testing graph compilation...")
    
    try:
        from agent.graph import graph
        print("  ✅ Graph compiled successfully")
        print(f"  - Graph name: {graph.name}")
        return True
    except Exception as e:
        print(f"  ❌ Graph compilation failed: {e}")
        return False

async def test_simple_invocation():
    """Test a simple graph invocation."""
    print("\n🔍 Testing simple graph invocation...")
    
    try:
        from agent.graph import graph
        from agent.state import OverallState, QueryType
        from langchain_core.messages import HumanMessage
        
        # Create a simple test state
        test_state: OverallState = {
            "messages": [HumanMessage(content="Test query")],
            "user_query": "Test query",
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
        
        print("  - Invoking graph with test state...")
        result = await graph.ainvoke(test_state)
        print("  ✅ Graph invocation successful")
        print(f"  - Result keys: {list(result.keys())}")
        return True
        
    except Exception as e:
        print(f"  ❌ Graph invocation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main debug function."""
    print("🚀 Multi-Agent System Debug")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check the error messages above.")
        return False
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment tests failed. Please set GEMINI_API_KEY.")
        return False
    
    # Test graph compilation
    if not test_graph_compilation():
        print("\n❌ Graph compilation failed. Please check the error messages above.")
        return False
    
    # Test simple invocation
    if not await test_simple_invocation():
        print("\n❌ Graph invocation failed. Please check the error messages above.")
        return False
    
    print("\n🎉 All tests passed! The system should work correctly.")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
