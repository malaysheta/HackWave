import os
import time
import asyncio
from typing import List

from agent.tools_and_schemas import (
    QueryClassification,
    DomainExpertAnalysis,
    UXUISpecialistAnalysis,
    TechnicalArchitectAnalysis,
    ModeratorAggregation,
    DebateAnalysis,
    QueryType,
    DebateCategory,
)
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from agent.state import (
    OverallState,
    DomainExpertState,
    UXUISpecialistState,
    TechnicalArchitectState,
    ModeratorState,
    DebateAnalysisState,
)
from agent.configuration import Configuration
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
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
async def classify_query(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that classifies user queries to determine routing to appropriate specialists.
    
    Analyzes the user query to determine whether it's a domain, UX/UI, technical, or general query,
    and routes it accordingly. Also handles debate detection and routing.
    
    Args:
        state: Current graph state containing the user query
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including query_type and debate_category
    """
    configurable = Configuration.from_runnable_config(config)
    start_time = time.time()
    
    # Initialize Gemini 2.0 Flash for query classification
    llm = ChatGoogleGenerativeAI(
        model=configurable.model,
        temperature=0.3,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(QueryClassification)
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_classification_instructions.format(
        user_query=state["user_query"],
        current_date=current_date,
    )
    
    # Classify the query using async execution
    result = await structured_llm.ainvoke(formatted_prompt)
    
    # Check if this is a debate (contains debate keywords)
    debate_keywords = ["debate", "conflict", "disagreement", "argument", "dispute", "controversy"]
    is_debate = any(keyword in state["user_query"].lower() for keyword in debate_keywords)
    
    if is_debate:
        # Route to debate analysis
        return {
            "query_type": QueryType.GENERAL,
            "debate_category": DebateCategory.MODERATOR,
            "processing_time": time.time() - start_time
        }
    
    return {
        "query_type": result.query_type,
        "debate_category": None,
        "processing_time": time.time() - start_time
    }


def route_to_specialists(state: OverallState) -> List[Send]:
    """LangGraph node that routes queries to appropriate specialist agents.
    
    Based on the query classification, routes to one or more specialist agents
    for parallel processing.
    
    Args:
        state: Current graph state containing the query type
        
    Returns:
        List of Send objects to route to appropriate specialists
    """
    routes = []
    
    # For general queries, route to all specialists
    if state["query_type"] == QueryType.GENERAL:
        routes.append(Send("domain_expert", {"user_query": state["user_query"]}))
        routes.append(Send("ux_ui_specialist", {"user_query": state["user_query"]}))
        routes.append(Send("technical_architect", {"user_query": state["user_query"]}))
    elif state["query_type"] == QueryType.DOMAIN:
        routes.append(Send("domain_expert", {"user_query": state["user_query"]}))
    elif state["query_type"] == QueryType.UX_UI:
        routes.append(Send("ux_ui_specialist", {"user_query": state["user_query"]}))
    elif state["query_type"] == QueryType.TECHNICAL:
        routes.append(Send("technical_architect", {"user_query": state["user_query"]}))
    else:
        # Default fallback - route to all specialists
        routes.append(Send("domain_expert", {"user_query": state["user_query"]}))
        routes.append(Send("ux_ui_specialist", {"user_query": state["user_query"]}))
        routes.append(Send("technical_architect", {"user_query": state["user_query"]}))
    
    return routes


async def domain_expert_analysis(state: DomainExpertState, config: RunnableConfig) -> OverallState:
    """LangGraph node for Domain Expert analysis.
    
    Analyzes product requirements from a business and domain perspective,
    focusing on business logic, industry standards, and domain-specific requirements.
    
    Args:
        state: Current graph state containing the user query
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update containing domain expert analysis
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize Gemini 2.0 Flash for domain expert analysis
    llm = ChatGoogleGenerativeAI(
        model=configurable.model,
        temperature=0.7,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(DomainExpertAnalysis)
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = domain_expert_instructions.format(
        user_query=state["user_query"],
        current_date=current_date,
    )
    
    # Generate domain expert analysis using async execution
    result = await structured_llm.ainvoke(formatted_prompt)
    
    return {
        "domain_expert_analysis": f"""
Domain Analysis: {result.domain_analysis}

Domain Requirements:
{chr(10).join(f"- {req}" for req in result.domain_requirements)}

Domain Concerns:
{chr(10).join(f"- {concern}" for concern in result.domain_concerns)}

Priority Level: {result.priority_level}
        """.strip()
    }


async def ux_ui_specialist_analysis(state: UXUISpecialistState, config: RunnableConfig) -> OverallState:
    """LangGraph node for UX/UI Specialist analysis.
    
    Analyzes product requirements from a user experience and interface design perspective,
    focusing on usability, accessibility, and user interaction patterns.
    
    Args:
        state: Current graph state containing the user query
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update containing UX/UI specialist analysis
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize Gemini 2.0 Flash for UX/UI specialist analysis
    llm = ChatGoogleGenerativeAI(
        model=configurable.model,
        temperature=0.7,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(UXUISpecialistAnalysis)
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = ux_ui_specialist_instructions.format(
        user_query=state["user_query"],
        current_date=current_date,
    )
    
    # Generate UX/UI specialist analysis using async execution
    result = await structured_llm.ainvoke(formatted_prompt)
    
    return {
        "ux_ui_specialist_analysis": f"""
UX Analysis: {result.ux_analysis}

UI Requirements:
{chr(10).join(f"- {req}" for req in result.ui_requirements)}

User Experience Concerns:
{chr(10).join(f"- {concern}" for concern in result.user_experience_concerns)}

Accessibility Requirements:
{chr(10).join(f"- {req}" for req in result.accessibility_requirements)}
        """.strip()
    }


async def technical_architect_analysis(state: TechnicalArchitectState, config: RunnableConfig) -> OverallState:
    """LangGraph node for Technical Architect analysis.
    
    Analyzes product requirements from a technical architecture perspective,
    focusing on system design, scalability, performance, and technical implementation.
    
    Args:
        state: Current graph state containing the user query
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update containing technical architect analysis
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize Gemini 2.0 Flash for technical architect analysis
    llm = ChatGoogleGenerativeAI(
        model=configurable.model,
        temperature=0.7,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(TechnicalArchitectAnalysis)
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = technical_architect_instructions.format(
        user_query=state["user_query"],
        current_date=current_date,
    )
    
    # Generate technical architect analysis using async execution
    result = await structured_llm.ainvoke(formatted_prompt)
    
    return {
        "technical_architect_analysis": f"""
Technical Analysis: {result.technical_analysis}

Technical Requirements:
{chr(10).join(f"- {req}" for req in result.technical_requirements)}

Technical Concerns:
{chr(10).join(f"- {concern}" for concern in result.technical_concerns)}

Scalability Considerations:
{chr(10).join(f"- {consideration}" for consideration in result.scalability_considerations)}
        """.strip()
    }


async def analyze_debate(state: DebateAnalysisState, config: RunnableConfig) -> OverallState:
    """LangGraph node for debate analysis and routing.
    
    Analyzes debate content to determine the most appropriate specialist
    to handle the resolution, with a focus on efficiency (under 2 minutes).
    
    Args:
        state: Current graph state containing the debate content
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update containing debate analysis and routing decision
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize Gemini 2.0 Flash for debate analysis
    llm = ChatGoogleGenerativeAI(
        model=configurable.model,
        temperature=0.5,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(DebateAnalysis)
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = debate_analysis_instructions.format(
        debate_content=state["debate_content"],
        user_query=state["user_query"],
        current_date=current_date,
    )
    
    # Generate debate analysis using async execution
    result = await structured_llm.ainvoke(formatted_prompt)
    
    return {
        "debate_category": result.debate_category,
        "debate_resolution": f"""
Debate Analysis:
- Category: {result.debate_category.value}
- Routing Decision: {result.routing_decision}
- Urgency Level: {result.urgency_level}
- Estimated Resolution Time: {result.estimated_resolution_time}
        """.strip()
    }


async def moderator_aggregation(state: ModeratorState, config: RunnableConfig) -> OverallState:
    """LangGraph node for Moderator/Aggregator analysis.
    
    Aggregates feedback from multiple specialist agents and resolves conflicts
    to create a unified product requirements specification.
    
    Args:
        state: Current graph state containing specialist analyses
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update containing moderator aggregation
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize Gemini 2.0 Flash for moderator aggregation
    llm = ChatGoogleGenerativeAI(
        model=configurable.model,
        temperature=0.5,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(ModeratorAggregation)
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = moderator_aggregation_instructions.format(
        domain_analysis=state.get("domain_expert_analysis", "No domain analysis provided"),
        ux_analysis=state.get("ux_ui_specialist_analysis", "No UX/UI analysis provided"),
        technical_analysis=state.get("technical_architect_analysis", "No technical analysis provided"),
        user_query=state["user_query"],
        current_date=current_date,
    )
    
    # Generate moderator aggregation using async execution
    result = await structured_llm.ainvoke(formatted_prompt)
    
    return {
        "moderator_aggregation": f"""
Aggregated Requirements:
{chr(10).join(f"- {req}" for req in result.aggregated_requirements)}

Conflict Resolution:
{result.conflict_resolution if result.conflict_resolution else "No conflicts identified"}

Final Recommendations:
{chr(10).join(f"- {rec}" for rec in result.final_recommendations)}

Implementation Priority:
{chr(10).join(f"- {priority}" for priority in result.implementation_priority)}
        """.strip()
    }


async def finalize_answer(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that finalizes the product requirements answer.
    
    Creates a comprehensive, well-structured final answer based on the
    aggregated specialist analyses and moderator aggregation.
    
    Args:
        state: Current graph state containing all analyses
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update containing the final answer
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize Gemini 2.0 Flash for final answer generation
    llm = ChatGoogleGenerativeAI(
        model=configurable.model,
        temperature=0.3,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = final_answer_instructions.format(
        user_query=state["user_query"],
        moderator_aggregation=state.get("moderator_aggregation", "No aggregation available"),
        current_date=current_date,
    )
    
    # Generate final answer using async execution
    result = await llm.ainvoke(formatted_prompt)
    
    return {
        "messages": [AIMessage(content=result.content)],
        "final_answer": result.content,
    }


# Create our Multi-Agent Product Requirements Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes
builder.add_node("classify_query", classify_query)
builder.add_node("domain_expert", domain_expert_analysis)
builder.add_node("ux_ui_specialist", ux_ui_specialist_analysis)
builder.add_node("technical_architect", technical_architect_analysis)
builder.add_node("analyze_debate", analyze_debate)
builder.add_node("moderator_aggregation", moderator_aggregation)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint
builder.add_edge(START, "classify_query")

# Add conditional edges for routing
builder.add_conditional_edges(
    "classify_query", 
    lambda state: "analyze_debate" if state.get("debate_category") else route_to_specialists(state),
    ["analyze_debate", "domain_expert", "ux_ui_specialist", "technical_architect"]
)

# Route debate analysis to appropriate specialist or moderator
builder.add_conditional_edges(
    "analyze_debate",
    lambda state: {
        DebateCategory.DOMAIN_EXPERT: "domain_expert",
        DebateCategory.UX_UI_SPECIALIST: "ux_ui_specialist", 
        DebateCategory.TECHNICAL_ARCHITECT: "technical_architect",
        DebateCategory.MODERATOR: "moderator_aggregation"
    }.get(state.get("debate_category"), "moderator_aggregation"),
    ["domain_expert", "ux_ui_specialist", "technical_architect", "moderator_aggregation"]
)

# Route specialist analyses to moderator aggregation
builder.add_edge("domain_expert", "moderator_aggregation")
builder.add_edge("ux_ui_specialist", "moderator_aggregation")
builder.add_edge("technical_architect", "moderator_aggregation")

# Route moderator aggregation to final answer
builder.add_edge("moderator_aggregation", "finalize_answer")

# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="multi-agent-product-requirements")
