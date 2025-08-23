#!/usr/bin/env python3
"""
CLI example for the Multi-Agent Product Requirements Refinement System.

This script demonstrates how to use the multi-agent system from the command line
to refine product requirements through specialized agent analysis.
"""

import sys
import os
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.graph import graph
from agent.state import OverallState, QueryType
from langchain_core.messages import HumanMessage


def main():
    """Main function to run the multi-agent product requirements refinement."""
    
    if len(sys.argv) < 2:
        print("Usage: python cli_research.py <product_requirements_query>")
        print("\nExample queries:")
        print("  'What are the requirements for a mobile banking app?'")
        print("  'Design requirements for an e-commerce platform'")
        print("  'Technical architecture for a real-time chat application'")
        print("  'UX/UI requirements for a healthcare management system'")
        sys.exit(1)
    
    # Get the query from command line arguments
    query = " ".join(sys.argv[1:])
    
    print("🤖 Multi-Agent Product Requirements Refinement System")
    print("=" * 60)
    print(f"Query: {query}")
    print("-" * 60)
    
    # Prepare the initial state
    initial_state: OverallState = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
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
    
    start_time = time.time()
    
    try:
        print("🔄 Processing query through multi-agent system...")
        print()
        
        # Run the graph
        result = graph.invoke(initial_state)
        
        total_time = time.time() - start_time
        
        # Display results
        print("📊 ANALYSIS RESULTS")
        print("=" * 60)
        
        # Query classification
        print(f"🔍 Query Type: {result.get('query_type', 'Unknown').value}")
        if result.get('debate_category'):
            print(f"⚖️  Debate Category: {result.get('debate_category').value}")
        print()
        
        # Domain Expert Analysis
        if result.get('domain_expert_analysis'):
            print("🏢 DOMAIN EXPERT ANALYSIS")
            print("-" * 30)
            print(result['domain_expert_analysis'])
            print()
        
        # UX/UI Specialist Analysis
        if result.get('ux_ui_specialist_analysis'):
            print("🎨 UX/UI SPECIALIST ANALYSIS")
            print("-" * 30)
            print(result['ux_ui_specialist_analysis'])
            print()
        
        # Technical Architect Analysis
        if result.get('technical_architect_analysis'):
            print("⚙️  TECHNICAL ARCHITECT ANALYSIS")
            print("-" * 30)
            print(result['technical_architect_analysis'])
            print()
        
        # Moderator Aggregation
        if result.get('moderator_aggregation'):
            print("🤝 MODERATOR AGGREGATION")
            print("-" * 30)
            print(result['moderator_aggregation'])
            print()
        
        # Debate Resolution
        if result.get('debate_resolution'):
            print("⚖️  DEBATE RESOLUTION")
            print("-" * 30)
            print(result['debate_resolution'])
            print()
        
        # Final Answer
        print("✅ FINAL ANSWER")
        print("=" * 60)
        final_answer = ""
        if result.get("messages"):
            for message in result["messages"]:
                if hasattr(message, 'content'):
                    final_answer = message.content
                    break
        
        if final_answer:
            print(final_answer)
        elif result.get("final_answer"):
            print(result["final_answer"])
        else:
            print("No final answer generated.")
        
        print()
        print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
