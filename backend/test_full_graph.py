#!/usr/bin/env python3
"""
Test script to build the full graph step by step.
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

def test_full_graph_build():
    """Test building the full graph step by step."""
    print("Testing full graph build...")
    
    try:
        print("1. Importing required modules...")
        from langgraph.graph import StateGraph, START, END
        from src.agent.state import OverallState
        from src.agent.configuration import Configuration
        from langgraph.checkpoint.memory import MemorySaver
        print("✅ Imports successful")
        
        print("2. Creating StateGraph...")
        builder = StateGraph(OverallState, context_schema=Configuration)
        print("✅ StateGraph created")
        
        print("3. Importing all nodes...")
        from src.agent.graph import (
            supervisor_node, classify_query, domain_expert_analysis,
            ux_ui_specialist_analysis, technical_architect_analysis,
            revenue_model_analyst_analysis, analyze_debate,
            moderator_aggregation, finalize_answer, supervisor_router
        )
        print("✅ All nodes imported")
        
        print("4. Adding nodes one by one...")
        builder.add_node("supervisor", supervisor_node)
        print("✅ Supervisor node added")
        
        builder.add_node("classify_query", classify_query)
        print("✅ Classify query node added")
        
        builder.add_node("domain_expert", domain_expert_analysis)
        print("✅ Domain expert node added")
        
        builder.add_node("ux_ui_specialist", ux_ui_specialist_analysis)
        print("✅ UX/UI specialist node added")
        
        builder.add_node("technical_architect", technical_architect_analysis)
        print("✅ Technical architect node added")
        
        builder.add_node("revenue_model_analyst", revenue_model_analyst_analysis)
        print("✅ Revenue model analyst node added")
        
        builder.add_node("analyze_debate", analyze_debate)
        print("✅ Analyze debate node added")
        
        builder.add_node("moderator_aggregation", moderator_aggregation)
        print("✅ Moderator aggregation node added")
        
        builder.add_node("finalize_answer", finalize_answer)
        print("✅ Finalize answer node added")
        
        print("5. Adding edges...")
        builder.add_edge(START, "classify_query")
        print("✅ Start edge added")
        
        builder.add_conditional_edges(
            "classify_query",
            lambda state: "supervisor",
            ["supervisor"]
        )
        print("✅ Classify query edges added")
        
        builder.add_conditional_edges(
            "supervisor",
            supervisor_router,
            ["domain_expert", "ux_ui_specialist", "technical_architect", "revenue_model_analyst", 
             "moderator_aggregation", "analyze_debate", "finalize_answer"]
        )
        print("✅ Supervisor edges added")
        
        builder.add_edge("domain_expert", "supervisor")
        builder.add_edge("ux_ui_specialist", "supervisor")
        builder.add_edge("technical_architect", "supervisor")
        builder.add_edge("revenue_model_analyst", "supervisor")
        builder.add_edge("moderator_aggregation", "supervisor")
        builder.add_edge("analyze_debate", "supervisor")
        print("✅ Agent return edges added")
        
        builder.add_edge("finalize_answer", END)
        print("✅ End edge added")
        
        print("6. Creating checkpoint saver...")
        checkpoint_saver = MemorySaver()
        print("✅ Checkpoint saver created")
        
        print("7. Compiling graph...")
        graph = builder.compile(
            name="supervisor-based-multi-agent-product-requirements",
            checkpointer=checkpoint_saver
        )
        print("✅ Graph compiled successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Full graph build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Full Graph Build Test")
    print("=" * 50)
    
    # Test full graph build
    if not test_full_graph_build():
        print("\n❌ Full graph build test failed")
        return False
    
    print("\n🎉 Full graph build test passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

