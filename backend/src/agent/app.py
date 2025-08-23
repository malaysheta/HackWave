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

from agent.graph import graph
from agent.state import OverallState, QueryType, DebateCategory


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


# Define the FastAPI app
app = FastAPI(
    title="Multi-Agent Product Requirements Refinement System",
    description="A sophisticated multi-agent AI system for refining product requirements with debate handling capabilities",
    version="1.0.0"
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
    """Stream the graph execution with real-time updates."""
    
    try:
        # Run the graph and capture results
        result = await graph.ainvoke(initial_state)
        
        # Stream each analysis with delays to simulate real-time updates
        if result.get("domain_expert_analysis"):
            await asyncio.sleep(0.5)  # Simulate processing time
            yield f"data: {json.dumps({'type': 'domain_expert', 'content': result['domain_expert_analysis']})}\n\n"
        
        if result.get("ux_ui_specialist_analysis"):
            await asyncio.sleep(0.5)  # Simulate processing time
            yield f"data: {json.dumps({'type': 'ux_ui_specialist', 'content': result['ux_ui_specialist_analysis']})}\n\n"
        
        if result.get("technical_architect_analysis"):
            await asyncio.sleep(0.5)  # Simulate processing time
            yield f"data: {json.dumps({'type': 'technical_architect', 'content': result['technical_architect_analysis']})}\n\n"
        
        if result.get("moderator_aggregation"):
            await asyncio.sleep(0.5)  # Simulate processing time
            yield f"data: {json.dumps({'type': 'moderator_aggregation', 'content': result['moderator_aggregation']})}\n\n"
        
        if result.get("final_answer"):
            await asyncio.sleep(0.5)  # Simulate processing time
            yield f"data: {json.dumps({'type': 'final_answer', 'content': result['final_answer']})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
    except Exception as e:
        print(f"Error in streaming: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


# API Endpoints
@app.post("/api/refine-requirements", response_model=ProductRequirementsResponse)
async def refine_product_requirements(request: ProductRequirementsRequest):
    """
    Refine product requirements using the multi-agent system.
    
    This endpoint processes product requirement queries through specialized agents:
    - Domain Expert: Analyzes business logic and domain-specific requirements
    - UX/UI Specialist: Handles user experience and interface design requirements
    - Technical Architect: Manages technical architecture and implementation requirements
    - Moderator/Aggregator: Consolidates feedback and resolves conflicts
    
    The system also includes debate handling capabilities for resolving conflicts efficiently.
    """
    try:
        start_time = time.time()
        
        # Prepare the initial state
        initial_state: OverallState = {
            "messages": [HumanMessage(content=request.query)],
            "user_query": request.query,
            "query_type": QueryType.GENERAL,  # Will be determined by classify_query node
            "debate_category": None,
            "domain_expert_analysis": None,
            "ux_ui_specialist_analysis": None,
            "technical_architect_analysis": None,
            "moderator_aggregation": None,
            "debate_resolution": None,
            "final_answer": None,
            "processing_time": 0.0
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
            debate_resolution=result.get("debate_resolution")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/api/refine-requirements/stream")
async def refine_product_requirements_stream(request: ProductRequirementsRequest):
    """
    Stream product requirements refinement using Server-Sent Events.
    
    This endpoint provides real-time streaming of the multi-agent analysis process,
    allowing the frontend to display progress updates as each specialist completes their analysis.
    """
    try:
        # Prepare the initial state
        initial_state: OverallState = {
            "messages": [HumanMessage(content=request.query)],
            "user_query": request.query,
            "query_type": QueryType.GENERAL,  # Will be determined by classify_query node
            "debate_category": None,
            "domain_expert_analysis": None,
            "ux_ui_specialist_analysis": None,
            "technical_architect_analysis": None,
            "moderator_aggregation": None,
            "debate_resolution": None,
            "final_answer": None,
            "processing_time": 0.0
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
    """Health check endpoint for the multi-agent system."""
    return {
        "status": "healthy",
        "system": "Multi-Agent Product Requirements Refinement System",
        "version": "1.0.0",
        "agents": [
            "Domain Expert",
            "UX/UI Specialist", 
            "Technical Architect",
            "Moderator/Aggregator",
            "Debate Handler"
        ]
    }


@app.get("/api/agents")
async def get_agents_info():
    """Get information about available specialist agents."""
    return {
        "agents": {
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
