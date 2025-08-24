# LangGraph MongoDB Memory Integration

This document explains how to use the MongoDB memory integration with your LangGraph agent system.

## Overview

The memory system provides:
- **Conversation History**: Persistent storage of all conversation steps
- **State Persistence**: Automatic saving and loading of graph state
- **Thread Isolation**: Each conversation thread has isolated memory
- **Context Awareness**: Agents can access previous conversation context
- **Checkpoint Management**: Automatic checkpoint saving for graph state

## Prerequisites

1. **MongoDB**: Make sure MongoDB is running on `localhost:27017`
2. **Python Dependencies**: Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Variables**: Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Quick Start

### 1. Test the Memory System

First, test that everything is working:

```bash
cd backend
python test_memory.py
```

This will verify:
- MongoDB connection
- Memory manager functionality
- Data persistence and retrieval
- Thread isolation

### 2. Run the Agent Demo

Run the full agent demo with memory persistence:

```bash
cd backend
python run_agent.py
```

This demonstrates:
- Multiple sessions with the same `thread_id`
- Memory persistence between sessions
- Context-aware responses
- New thread isolation

## Usage Examples

### Basic Usage

```python
import asyncio
from src.agent.graph import graph
from src.agent.state import OverallState

async def run_agent_with_memory():
    # Prepare initial state
    initial_state: OverallState = {
        "messages": [],
        "user_query": "I want to build a mobile app for food delivery",
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
        "max_steps": 10,
        "is_complete": False
    }
    
    # Configuration with thread_id for memory persistence
    config = {
        "configurable": {
            "model": "gemini-2.0-flash",
            "thread_id": "my_conversation_001",  # Unique thread ID
            "max_debate_resolution_time": 120,
            "enable_parallel_processing": True
        }
    }
    
    # Run the graph
    result = await graph.ainvoke(initial_state, config)
    return result

# Run the agent
result = asyncio.run(run_agent_with_memory())
print(result.get("final_answer"))
```

### Memory Manager Direct Usage

```python
from src.agent.memory import create_memory_manager

# Create memory manager
memory_manager = create_memory_manager()

# Save conversation memory
thread_id = "my_thread_001"
state = {
    "user_query": "How should I monetize my app?",
    "current_step": 1,
    "agent_history": [...],
    "final_answer": "Consider freemium model..."
}

success = memory_manager.save_conversation_memory(thread_id, state)

# Retrieve conversation history
history = memory_manager.get_conversation_history(thread_id, limit=10)

# Get thread summary
summary = memory_manager.get_thread_summary(thread_id)
print(f"Thread has {summary['conversation_count']} conversations")

# Clear thread memory
memory_manager.clear_thread_memory(thread_id)
```

## Architecture

### Memory Components

1. **MongoDBMemoryManager**: Main memory management class
   - Handles conversation history
   - Manages memory context
   - Provides thread isolation

2. **MongoDBCheckpointSaver**: Checkpoint persistence
   - Saves graph state checkpoints
   - Enables state restoration
   - Integrates with LangGraph checkpoint system

3. **Database Collections**:
   - `conversations`: Conversation history and state snapshots
   - `checkpoints`: Graph state checkpoints
   - `memory_context`: Additional context data

### Integration Points

1. **Graph Nodes**: Each node can access and save memory
2. **Configuration**: `thread_id` is passed through `RunnableConfig`
3. **State Management**: Automatic state persistence and restoration
4. **Context Injection**: Previous conversation context is injected into prompts

## Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your-gemini-api-key

# Optional (defaults shown)
MONGODB_URL=mongodb://localhost:27017/Hackwave
```

### Configuration Schema

```python
class Configuration(BaseModel):
    model: str = "gemini-2.0-flash"
    thread_id: Optional[str] = None  # For memory persistence
    max_debate_resolution_time: int = 120
    enable_parallel_processing: bool = True
```

## Memory Features

### Conversation History

- **Automatic Saving**: Each agent step saves to memory
- **Context Retrieval**: Previous conversations are retrieved for context
- **Thread Isolation**: Each `thread_id` has isolated memory
- **Performance Indexes**: Optimized database queries

### State Persistence

- **Checkpoint Saving**: Automatic graph state checkpointing
- **State Restoration**: Can resume from previous state
- **Error Recovery**: State is preserved across errors

### Context Awareness

- **Previous Queries**: Agents can see previous user queries
- **Agent Decisions**: Previous agent decisions are available
- **Analysis Results**: Previous analysis results are accessible

## Best Practices

### Thread ID Management

1. **Use Unique IDs**: Each conversation should have a unique `thread_id`
2. **Consistent Naming**: Use consistent naming conventions for thread IDs
3. **Session Management**: Associate thread IDs with user sessions

### Memory Cleanup

1. **Regular Cleanup**: Periodically clean up old threads
2. **Size Limits**: Monitor database size and implement retention policies
3. **Error Handling**: Handle memory operation failures gracefully

### Performance Optimization

1. **Index Usage**: Database indexes are automatically created
2. **Query Limits**: Use limits when retrieving history
3. **Connection Pooling**: MongoDB connection is reused

## Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   - Ensure MongoDB is running: `mongod`
   - Check connection URL in memory manager
   - Verify network connectivity

2. **Memory Not Persisting**
   - Check that `thread_id` is provided in configuration
   - Verify MongoDB write permissions
   - Check for exceptions in memory operations

3. **Context Not Available**
   - Ensure conversation history exists for the thread
   - Check that memory is being saved correctly
   - Verify thread ID consistency

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check memory operations:

```python
# Test memory manager directly
from src.agent.memory import create_memory_manager
memory_manager = create_memory_manager()

# Test basic operations
success = memory_manager.save_conversation_memory("test", {"test": "data"})
print(f"Save success: {success}")

history = memory_manager.get_conversation_history("test")
print(f"History: {history}")
```

## API Reference

### MongoDBMemoryManager

```python
class MongoDBMemoryManager:
    def __init__(self, mongodb_url: str = "mongodb://localhost:27017/Hackwave")
    def save_conversation_memory(self, thread_id: str, state: Dict[str, Any]) -> bool
    def get_conversation_history(self, thread_id: str, limit: int = 10) -> List[Dict[str, Any]]
    def save_memory_context(self, thread_id: str, context: Dict[str, Any]) -> bool
    def get_memory_context(self, thread_id: str) -> Optional[Dict[str, Any]]
    def get_thread_summary(self, thread_id: str) -> Dict[str, Any]
    def clear_thread_memory(self, thread_id: str) -> bool
    def close(self)
```

### MongoDBCheckpointSaver

```python
class MongoDBCheckpointSaver(BaseCheckpointSaver):
    def __init__(self, mongodb_url: str = "mongodb://localhost:27017/Hackwave")
    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]
    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> None
    def close(self)
```

## Examples

See the following files for complete examples:
- `run_agent.py`: Full agent demo with memory
- `test_memory.py`: Memory system testing
- `src/agent/graph.py`: Integration in graph nodes

