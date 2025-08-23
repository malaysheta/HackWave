from typing import List
from langchain_core.messages import AnyMessage, HumanMessage


def get_user_query(messages: List[AnyMessage]) -> str:
    """
    Extract the user query from the messages.
    
    Args:
        messages: List of messages from the conversation
        
    Returns:
        The user's query as a string
    """
    if len(messages) == 1:
        return messages[-1].content
    
    # For multi-turn conversations, combine all user messages
    user_query = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            user_query += f"{message.content}\n"
    
    return user_query.strip()


def format_agent_response(agent_name: str, analysis: str, requirements: List[str], concerns: List[str]) -> str:
    """
    Format an agent's response in a consistent way.
    
    Args:
        agent_name: Name of the agent (e.g., "Domain Expert")
        analysis: The agent's analysis text
        requirements: List of requirements identified
        concerns: List of concerns identified
        
    Returns:
        Formatted response string
    """
    formatted = f"{agent_name} Analysis:\n{analysis}\n\n"
    
    if requirements:
        formatted += "Requirements:\n"
        for req in requirements:
            formatted += f"- {req}\n"
        formatted += "\n"
    
    if concerns:
        formatted += "Concerns:\n"
        for concern in concerns:
            formatted += f"- {concern}\n"
        formatted += "\n"
    
    return formatted.strip()
