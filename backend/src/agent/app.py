# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
import time
import asyncio
import json
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import FastAPI, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from src.agent.graph import graph
from src.agent.state import OverallState, QueryType, DebateCategory, AgentType, SupervisorDecision


# Define request/response models
class ProductRequirementsRequest(BaseModel):
    query: str
    query_type: Optional[str] = None
    debate_content: Optional[str] = None


class ProductRequirementsResponse(BaseModel):
    answer: str
    processing_time: float
    query_type: str
    debate_category: Optional[str] = None
    domain_analysis: Optional[str] = None
    ux_analysis: Optional[str] = None
    technical_analysis: Optional[str] = None
    moderator_aggregation: Optional[str] = None
    debate_resolution: Optional[str] = None
    agent_history: Optional[list] = None
    supervisor_reasoning: Optional[str] = None


# Define the FastAPI app
app = FastAPI(
    title="Supervisor-Based Multi-Agent Product Requirements Refinement System",
    description="A sophisticated supervisor-based multi-agent AI system for refining product requirements with dynamic routing and debate handling capabilities",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # Return a dummy router if build isn't ready
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


async def stream_graph_execution(initial_state: OverallState) -> AsyncGenerator[str, None]:
    """Stream the graph execution with real-time updates for Supervisor-based architecture."""
    
    try:
        # Run the graph and capture results
        result = await graph.ainvoke(initial_state)
        
        # Stream supervisor decisions and agent activities
        agent_history = result.get("agent_history", [])
        
        for entry in agent_history:
            await asyncio.sleep(0.3)  # Simulate processing time
            
            if entry.get("agent") == "supervisor":
                yield f"data: {json.dumps({'type': 'supervisor_decision', 'content': entry.get('reasoning', 'Supervisor analyzing...')})}\n\n"
            elif entry.get("agent") == "domain_expert":
                yield f"data: {json.dumps({'type': 'domain_expert', 'content': result.get('domain_expert_analysis', 'Domain analysis completed')})}\n\n"
            elif entry.get("agent") == "ux_ui_specialist":
                yield f"data: {json.dumps({'type': 'ux_ui_specialist', 'content': result.get('ux_ui_specialist_analysis', 'UX/UI analysis completed')})}\n\n"
            elif entry.get("agent") == "technical_architect":
                yield f"data: {json.dumps({'type': 'technical_architect', 'content': result.get('technical_architect_analysis', 'Technical analysis completed')})}\n\n"
            elif entry.get("agent") == "revenue_model_analyst":
                yield f"data: {json.dumps({'type': 'revenue_model_analyst', 'content': result.get('revenue_model_analyst_analysis', 'Revenue analysis completed')})}\n\n"
            elif entry.get("agent") == "moderator":
                yield f"data: {json.dumps({'type': 'moderator_aggregation', 'content': result.get('moderator_aggregation', 'Moderator aggregation completed')})}\n\n"
            elif entry.get("agent") == "debate_analyzer":
                yield f"data: {json.dumps({'type': 'debate_analysis', 'content': result.get('debate_resolution', 'Debate analysis completed')})}\n\n"
            elif entry.get("agent") == "finalizer":
                yield f"data: {json.dumps({'type': 'final_answer', 'content': result.get('final_answer', 'Final answer generated')})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
    except Exception as e:
        print(f"Error in streaming: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


# API Endpoints
@app.post("/api/refine-requirements", response_model=ProductRequirementsResponse)
async def refine_product_requirements(request: ProductRequirementsRequest):
    """
    Refine product requirements using the supervisor-based multi-agent system.
    
    This endpoint processes product requirement queries through a Supervisor that coordinates:
    - Domain Expert: Analyzes business logic and domain-specific requirements
    - UX/UI Specialist: Handles user experience and interface design requirements
    - Technical Architect: Manages technical architecture and implementation requirements
    - Revenue Model Analyst: Analyzes revenue models and monetization strategies
    - Moderator/Aggregator: Consolidates feedback and resolves conflicts
    
    The Supervisor dynamically routes queries and handles debate resolution efficiently.
    """
    try:
        start_time = time.time()
        
        # Prepare the initial state with Supervisor-related fields
        initial_state: OverallState = {
            "messages": [HumanMessage(content=request.query)],
            "user_query": request.query,
            "query_type": QueryType.GENERAL,  # Will be determined by classify_query node
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
        
        # If debate content is provided, add it to the state
        if request.debate_content:
            initial_state["debate_content"] = request.debate_content
            initial_state["debate_category"] = DebateCategory.MODERATOR
        
        # Run the graph using async execution
        result = await graph.ainvoke(initial_state)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Extract the final answer from messages
        final_answer = ""
        if result.get("messages"):
            for message in result["messages"]:
                if hasattr(message, 'content'):
                    final_answer = message.content
                    break
        
        return ProductRequirementsResponse(
            answer=final_answer or result.get("final_answer", "No answer generated"),
            processing_time=total_time,
            query_type=result.get("query_type", QueryType.GENERAL).value,
            debate_category=result.get("debate_category", DebateCategory.MODERATOR).value if result.get("debate_category") else None,
            domain_analysis=result.get("domain_expert_analysis"),
            ux_analysis=result.get("ux_ui_specialist_analysis"),
            technical_analysis=result.get("technical_architect_analysis"),
            moderator_aggregation=result.get("moderator_aggregation"),
            debate_resolution=result.get("debate_resolution"),
            agent_history=result.get("agent_history"),
            supervisor_reasoning=result.get("supervisor_reasoning")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/api/refine-requirements/stream")
async def refine_product_requirements_stream(request: ProductRequirementsRequest):
    """
    Stream product requirements refinement using Server-Sent Events.
    
    This endpoint provides real-time streaming of the supervisor-based multi-agent analysis process,
    allowing the frontend to display progress updates as the Supervisor coordinates each specialist.
    """
    try:
        # Prepare the initial state with Supervisor-related fields
        initial_state: OverallState = {
            "messages": [HumanMessage(content=request.query)],
            "user_query": request.query,
            "query_type": QueryType.GENERAL,  # Will be determined by classify_query node
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
        
        # If debate content is provided, add it to the state
        if request.debate_content:
            initial_state["debate_content"] = request.debate_content
            initial_state["debate_category"] = DebateCategory.MODERATOR
        
        return StreamingResponse(
            stream_graph_execution(initial_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint for the supervisor-based multi-agent system."""
    return {
        "status": "healthy",
        "system": "Supervisor-Based Multi-Agent Product Requirements Refinement System",
        "version": "2.0.0",
        "architecture": "Supervisor-based with dynamic routing",
        "agents": [
            "Supervisor (Orchestrator)",
            "Domain Expert",
            "UX/UI Specialist", 
            "Technical Architect",
            "Revenue Model Analyst",
            "Moderator/Aggregator",
            "Debate Handler"
        ]
    }


@app.get("/api/agents")
async def get_agents_info():
    """Get information about available specialist agents and the Supervisor."""
    return {
        "agents": {
            "supervisor": {
                "name": "Supervisor (Orchestrator)",
                "description": "Coordinates and directs the workflow by deciding which specialist agent should act next",
                "expertise": ["Workflow Orchestration", "Dynamic Routing", "Decision Making", "Agent Coordination", "State Management"]
            },
            "domain_expert": {
                "name": "Domain Expert",
                "description": "Analyzes business logic, industry standards, compliance requirements, and domain-specific knowledge",
                "expertise": ["Business Logic", "Industry Standards", "Compliance", "Market Analysis", "Domain Knowledge"]
            },
            "ux_ui_specialist": {
                "name": "UX/UI Specialist", 
                "description": "Analyzes user experience requirements, interface design, accessibility, and usability",
                "expertise": ["User Experience", "Interface Design", "Accessibility", "Usability", "User Research"]
            },
            "technical_architect": {
                "name": "Technical Architect",
                "description": "Analyzes technical architecture, system design, scalability, and implementation requirements",
                "expertise": ["System Architecture", "Technology Stack", "Scalability", "Performance", "Security"]
            },
            "revenue_model_analyst": {
                "name": "Revenue Model Analyst",
                "description": "Analyzes revenue models, monetization strategies, pricing, and financial sustainability",
                "expertise": ["Revenue Models", "Monetization", "Pricing Strategies", "Business Models", "Financial Analysis"]
            },
            "moderator": {
                "name": "Moderator/Aggregator",
                "description": "Aggregates feedback from specialists and resolves conflicts to create unified requirements",
                "expertise": ["Conflict Resolution", "Requirements Aggregation", "Priority Setting", "Stakeholder Coordination"]
            },
            "debate_handler": {
                "name": "Debate Handler",
                "description": "Analyzes and routes debates to appropriate specialists for efficient resolution (under 2 minutes)",
                "expertise": ["Debate Analysis", "Conflict Routing", "Efficiency Optimization", "Specialist Coordination"]
            }
        }
    }


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
