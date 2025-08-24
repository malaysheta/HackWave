#!/usr/bin/env python3
"""
Setup script for LangGraph MongoDB Memory Integration.

This script helps you:
1. Verify MongoDB connection
2. Install required dependencies
3. Test the memory system
4. Set up initial configuration
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "pymongo",
        "langgraph",
        "langchain",
        "langchain-google-genai",
        "google-genai",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    return True

def check_mongodb():
    """Check if MongoDB is running and accessible."""
    print("\n🗄️  Checking MongoDB connection...")
    
    try:
        from pymongo import MongoClient
        
        # Try to connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/Hackwave", serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        
        # Test database access
        db = client.get_database()
        collections = db.list_collection_names()
        
        print("✅ MongoDB connection successful")
        print(f"   Database: Hackwave")
        print(f"   Collections: {collections}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n💡 To fix this:")
        print("   1. Install MongoDB: https://docs.mongodb.com/manual/installation/")
        print("   2. Start MongoDB: mongod")
        print("   3. Or use Docker: docker run -d -p 27017:27017 --name mongodb mongo:latest")
        return False

def check_environment():
    """Check environment variables."""
    print("\n🔑 Checking environment variables...")
    
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env file found and loaded")
    
    # Check GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY is set")
        # Mask the key for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"   Key: {masked_key}")
    else:
        print("❌ GEMINI_API_KEY is not set")
        print("\n💡 To fix this:")
        print("   1. Get a Gemini API key from: https://makersuite.google.com/app/apikey")
        print("   2. Add to .env file: GEMINI_API_KEY=your-key-here")
        print("   3. Or set environment variable: export GEMINI_API_KEY=your-key-here")
        return False
    
    return True

def test_memory_system():
    """Test the memory system functionality."""
    print("\n🧪 Testing memory system...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.agent.memory import create_memory_manager
        
        # Create memory manager
        memory_manager = create_memory_manager()
        
        # Test basic operations
        test_thread_id = "setup_test_001"
        test_state = {
            "user_query": "Setup test query",
            "current_step": 1,
            "agent_history": [],
            "is_complete": False
        }
        
        # Test save
        success = memory_manager.save_conversation_memory(test_thread_id, test_state)
        if not success:
            print("❌ Failed to save conversation memory")
            return False
        
        # Test retrieve
        history = memory_manager.get_conversation_history(test_thread_id, limit=5)
        if not history:
            print("❌ Failed to retrieve conversation history")
            return False
        
        # Test summary
        summary = memory_manager.get_thread_summary(test_thread_id)
        if not summary:
            print("❌ Failed to get thread summary")
            return False
        
        # Cleanup
        memory_manager.clear_thread_memory(test_thread_id)
        memory_manager.close()
        
        print("✅ Memory system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    env_file = Path(".env")
    if not env_file.exists():
        print("\n📝 Creating sample .env file...")
        
        sample_content = """# Gemini API Configuration
GEMINI_API_KEY=your-gemini-api-key-here

# MongoDB Configuration (optional)
MONGODB_URL=mongodb://localhost:27017/Hackwave

# LangGraph Configuration (optional)
LANGGRAPH_API_KEY=your-langgraph-api-key-here
"""
        
        with open(env_file, 'w') as f:
            f.write(sample_content)
        
        print("✅ Sample .env file created")
        print("   Please edit .env and add your actual API keys")
    else:
        print("\n✅ .env file already exists")

def run_quick_test():
    """Run a quick test of the agent system."""
    print("\n🚀 Running quick agent test...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.agent.graph import graph
        from src.agent.state import OverallState
        
        # Simple test state
        test_state: OverallState = {
            "messages": [],
            "user_query": "Test query for setup verification",
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
            "max_steps": 3,  # Short test
            "is_complete": False
        }
        
        config = {
            "configurable": {
                "model": "gemini-2.0-flash",
                "thread_id": "setup_test_002",
                "max_debate_resolution_time": 120,
                "enable_parallel_processing": True
            }
        }
        
        import asyncio
        result = asyncio.run(graph.ainvoke(test_state, config))
        
        if result.get("final_answer"):
            print("✅ Agent test completed successfully")
            print(f"   Final answer length: {len(result.get('final_answer', ''))} characters")
            return True
        else:
            print("❌ Agent test failed - no final answer generated")
            return False
            
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🔧 LangGraph MongoDB Memory Integration Setup")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 6
    
    # Check Python version
    if check_python_version():
        checks_passed += 1
    
    # Check dependencies
    if check_dependencies():
        checks_passed += 1
    
    # Check MongoDB
    if check_mongodb():
        checks_passed += 1
    
    # Check environment
    if check_environment():
        checks_passed += 1
    
    # Create sample .env if needed
    create_sample_env()
    checks_passed += 1
    
    # Test memory system
    if test_memory_system():
        checks_passed += 1
    
    # Run quick agent test
    if run_quick_test():
        checks_passed += 1
    
    # Summary
    print(f"\n📊 Setup Summary: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the full demo: python run_agent.py")
        print("2. Test memory system: python test_memory.py")
        print("3. Read documentation: MEMORY_INTEGRATION.md")
        return True
    else:
        print(f"\n⚠️  Setup incomplete. {total_checks - checks_passed} check(s) failed.")
        print("Please fix the issues above and run setup again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

