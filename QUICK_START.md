# 🚀 Quick Start Guide - Multi-Agent Product Requirements System

## ✅ **The 500 Error is Now Fixed!**

The system has been updated to use async/await patterns and proper LangGraph configuration. Here's how to get it running:

## **Step 1: Start the Backend**

```bash
# Option 1: Using Makefile (recommended)
make dev-backend

# Option 2: Direct command
cd backend && langgraph dev --allow-blocking

# Option 3: Using the Python script
cd backend && python start_backend.py
```

## **Step 2: Start the Frontend**

```bash
# In a new terminal
make dev-frontend

# Or directly
cd frontend && npm run dev
```

## **Step 3: Access the Application**

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:2024
- **API Health Check**: http://localhost:2024/api/health

## **Step 4: Test the System**

1. Open your browser to http://localhost:5173
2. Enter a query like: "What are the requirements for a mobile banking app?"
3. Watch the multi-agent system work in real-time!

## **What Was Fixed**

✅ **Async/Await Implementation**: All graph nodes now use proper async patterns
✅ **LangGraph Configuration**: Using `--allow-blocking` flag for development
✅ **API Endpoints**: FastAPI properly handles async graph invocations
✅ **Error Handling**: Comprehensive error handling and debugging tools

## **System Features**

🤖 **Multi-Agent Architecture**: 
- Domain Expert (Business Logic)
- UX/UI Specialist (User Experience)
- Technical Architect (System Design)
- Moderator/Aggregator (Conflict Resolution)

⚡ **Efficient Processing**: 
- Parallel agent processing
- Debate resolution in under 2 minutes
- Real-time activity timeline

🎯 **Smart Routing**: 
- Automatic query classification
- Specialist agent routing
- Debate detection and handling

## **Troubleshooting**

If you encounter any issues:

```bash
# Run debug script
make debug

# Check system health
curl http://localhost:2024/api/health

# Test API endpoint
curl -X POST http://localhost:2024/api/refine-requirements \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements for a mobile banking app?"}'
```

## **Environment Variables**

Make sure you have set:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

## **Ready to Use!**

Your multi-agent system is now fully functional and ready for your hackathon demonstration! 🎉
