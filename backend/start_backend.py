#!/usr/bin/env python3
"""
Simple script to start the backend with proper flags for development.
"""

import subprocess
import sys
import os

def main():
    """Start the LangGraph backend with blocking allowed."""
    print("🚀 Starting Multi-Agent Backend with blocking allowed...")
    print("📍 Backend will be available at: http://localhost:2024")
    print("🔗 API endpoints:")
    print("  - POST /api/refine-requirements")
    print("  - GET /api/health")
    print("  - GET /api/agents")
    print("")
    
    try:
        # Start langgraph dev with --allow-blocking flag
        subprocess.run([
            sys.executable, "-m", "langgraph", "dev", "--allow-blocking"
        ], cwd=os.path.dirname(__file__))
    except KeyboardInterrupt:
        print("\n👋 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
