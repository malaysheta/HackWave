import os
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBMemoryManager:
    """
    MongoDB-based memory manager for LangGraph applications.
    Handles conversation history, state persistence, and checkpoint management.
    """
    
    def __init__(self, mongodb_url: str = "mongodb://localhost:27017/Hackwave"):
        """
        Initialize the MongoDB memory manager.
        
        Args:
            mongodb_url: MongoDB connection URL
        """
        self.mongodb_url = mongodb_url
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.conversations: Optional[Collection] = None
        self.checkpoints: Optional[Collection] = None
        self.memory_context: Optional[Collection] = None
        
        self._connect()
        self._setup_indexes()
    
    def _connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.mongodb_url)
            self.db = self.client.get_database()
            self.conversations = self.db.conversations
            self.checkpoints = self.db.checkpoints
            self.memory_context = self.db.memory_context
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.mongodb_url}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _setup_indexes(self):
        """Setup database indexes for performance."""
        try:
            # Indexes for conversations collection
            self.conversations.create_index("thread_id")
            self.conversations.create_index("timestamp")
            self.conversations.create_index([("thread_id", 1), ("timestamp", -1)])
            
            # Indexes for checkpoints collection
            self.checkpoints.create_index("thread_id")
            self.checkpoints.create_index("checkpoint_id")
            self.checkpoints.create_index([("thread_id", 1), ("timestamp", -1)])
            
            # Indexes for memory context collection
            self.memory_context.create_index("thread_id")
            self.memory_context.create_index("timestamp")
            self.memory_context.create_index([("thread_id", 1), ("timestamp", -1)])
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to setup indexes: {e}")
            raise
    
    def _serialize_enum(self, obj):
        """Convert enum objects to strings for MongoDB serialization."""
        if hasattr(obj, 'value'):
            return obj.value
        return obj
    
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for MongoDB storage."""
        serialized = {}
        for key, value in state.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_state(value)
            elif isinstance(value, list):
                serialized[key] = [self._serialize_enum(item) if hasattr(item, 'value') else item for item in value]
            else:
                serialized[key] = self._serialize_enum(value)
        return serialized
    
    def save_conversation_memory(self, thread_id: str, state: Dict[str, Any]) -> bool:
        """
        Save conversation memory for a specific thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            state: Current state to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Serialize state for MongoDB storage
            serialized_state = self._serialize_state(state)
            
            # Prepare conversation data
            conversation_data = {
                "thread_id": thread_id,
                "timestamp": datetime.utcnow(),
                "user_query": serialized_state.get("user_query", ""),
                "current_step": serialized_state.get("current_step", 1),
                "agent_history": serialized_state.get("agent_history", []),
                "active_agent": serialized_state.get("active_agent", None),
                "supervisor_decision": serialized_state.get("supervisor_decision", None),
                "supervisor_reasoning": serialized_state.get("supervisor_reasoning", None),
                "is_complete": serialized_state.get("is_complete", False),
                "processing_time": serialized_state.get("processing_time", 0.0),
                "state_snapshot": {
                    "domain_expert_analysis": serialized_state.get("domain_expert_analysis"),
                    "ux_ui_specialist_analysis": serialized_state.get("ux_ui_specialist_analysis"),
                    "technical_architect_analysis": serialized_state.get("technical_architect_analysis"),
                    "revenue_model_analyst_analysis": serialized_state.get("revenue_model_analyst_analysis"),
                    "moderator_aggregation": serialized_state.get("moderator_aggregation"),
                    "debate_resolution": serialized_state.get("debate_resolution"),
                    "final_answer": serialized_state.get("final_answer"),
                }
            }
            
            # Insert into conversations collection
            result = self.conversations.insert_one(conversation_data)
            logger.info(f"Saved conversation memory for thread {thread_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation memory: {e}")
            return False
    
    def get_conversation_history(self, thread_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a specific thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            limit: Maximum number of history entries to retrieve
            
        Returns:
            List of conversation history entries
        """
        try:
            cursor = self.conversations.find(
                {"thread_id": thread_id},
                {"_id": 0}  # Exclude MongoDB _id field
            ).sort("timestamp", -1).limit(limit)
            
            history = list(cursor)
            logger.info(f"Retrieved {len(history)} conversation history entries for thread {thread_id}")
            return history
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []
    
    def save_memory_context(self, thread_id: str, context: Dict[str, Any]) -> bool:
        """
        Save memory context for a specific thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            context: Context data to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            context_data = {
                "thread_id": thread_id,
                "timestamp": datetime.utcnow(),
                "context": context
            }
            
            result = self.memory_context.insert_one(context_data)
            logger.info(f"Saved memory context for thread {thread_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memory context: {e}")
            return False
    
    def get_memory_context(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest memory context for a specific thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            
        Returns:
            Latest memory context or None if not found
        """
        try:
            context = self.memory_context.find_one(
                {"thread_id": thread_id},
                {"_id": 0, "context": 1}
            )
            
            if context:
                logger.info(f"Retrieved memory context for thread {thread_id}")
                return context.get("context")
            else:
                logger.info(f"No memory context found for thread {thread_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve memory context: {e}")
            return None
    
    def get_thread_summary(self, thread_id: str) -> Dict[str, Any]:
        """
        Get a summary of a conversation thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            
        Returns:
            Dictionary containing thread summary
        """
        try:
            # Get conversation count
            conversation_count = self.conversations.count_documents({"thread_id": thread_id})
            
            # Get latest conversation
            latest_conversation = self.conversations.find_one(
                {"thread_id": thread_id},
                sort=[("timestamp", -1)]
            )
            
            # Get memory context
            memory_context = self.get_memory_context(thread_id)
            
            summary = {
                "thread_id": thread_id,
                "conversation_count": conversation_count,
                "latest_conversation": latest_conversation,
                "memory_context": memory_context,
                "last_updated": latest_conversation.get("timestamp") if latest_conversation else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get thread summary: {e}")
            return {"thread_id": thread_id, "error": str(e)}
    
    def clear_thread_memory(self, thread_id: str) -> bool:
        """
        Clear all memory for a specific thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            # Delete from all collections
            conversations_result = self.conversations.delete_many({"thread_id": thread_id})
            checkpoints_result = self.checkpoints.delete_many({"thread_id": thread_id})
            memory_result = self.memory_context.delete_many({"thread_id": thread_id})
            
            logger.info(f"Cleared memory for thread {thread_id}: "
                       f"{conversations_result.deleted_count} conversations, "
                       f"{checkpoints_result.deleted_count} checkpoints, "
                       f"{memory_result.deleted_count} memory contexts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear thread memory: {e}")
            return False
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


class MongoDBCheckpointSaver(BaseCheckpointSaver):
    """
    MongoDB-based checkpoint saver for LangGraph.
    Extends the base checkpoint saver to use MongoDB for persistence.
    """
    
    def __init__(self, mongodb_url: str = "mongodb://localhost:27017/Hackwave"):
        """
        Initialize the MongoDB checkpoint saver.
        
        Args:
            mongodb_url: MongoDB connection URL
        """
        super().__init__()
        self.mongodb_url = mongodb_url
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.checkpoints: Optional[Collection] = None
        
        self._connect()
        self._setup_indexes()
    
    def _connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.mongodb_url)
            self.db = self.client.get_database()
            self.checkpoints = self.db.checkpoints
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB for checkpoints: {self.mongodb_url}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB for checkpoints: {e}")
            raise
    
    def _setup_indexes(self):
        """Setup database indexes for checkpoints."""
        try:
            self.checkpoints.create_index("thread_id")
            self.checkpoints.create_index("checkpoint_id")
            self.checkpoints.create_index([("thread_id", 1), ("timestamp", -1)])
            logger.info("Checkpoint indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to setup checkpoint indexes: {e}")
            raise
    
    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get a checkpoint by thread_id.
        
        Args:
            config: Configuration containing thread_id
            
        Returns:
            Checkpoint data or None if not found
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return None
            
            checkpoint = self.checkpoints.find_one(
                {"thread_id": thread_id},
                sort=[("timestamp", -1)]
            )
            
            if checkpoint:
                logger.info(f"Retrieved checkpoint for thread {thread_id}")
                return checkpoint.get("checkpoint_data")
            else:
                logger.info(f"No checkpoint found for thread {thread_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get checkpoint: {e}")
            return None
    
    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> None:
        """
        Save a checkpoint.
        
        Args:
            config: Configuration containing thread_id
            checkpoint: Checkpoint data to save
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                logger.warning("No thread_id provided for checkpoint")
                return
            
            checkpoint_data = {
                "thread_id": thread_id,
                "checkpoint_id": f"{thread_id}_{int(time.time())}",
                "timestamp": datetime.utcnow(),
                "checkpoint_data": checkpoint
            }
            
            result = self.checkpoints.insert_one(checkpoint_data)
            logger.info(f"Saved checkpoint for thread {thread_id}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB checkpoint connection closed")


def create_mongodb_checkpoint_saver(mongodb_url: str = "mongodb://localhost:27017/Hackwave") -> MongoDBCheckpointSaver:
    """
    Factory function to create a MongoDB checkpoint saver.
    
    Args:
        mongodb_url: MongoDB connection URL
        
    Returns:
        MongoDBCheckpointSaver instance
    """
    return MongoDBCheckpointSaver(mongodb_url)


def create_memory_manager(mongodb_url: str = "mongodb://localhost:27017/Hackwave") -> MongoDBMemoryManager:
    """
    Factory function to create a MongoDB memory manager.
    
    Args:
        mongodb_url: MongoDB connection URL
        
    Returns:
        MongoDBMemoryManager instance
    """
    return MongoDBMemoryManager(mongodb_url)
