# Setup Guide for Multi-Agent Product Requirements System

## Prerequisites

1. **Python 3.11+** installed on your system
2. **Node.js and npm** installed on your system
3. **Google Gemini API Key** - Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Quick Setup

### 1. Install Dependencies

```bash
# Install all dependencies
make install
```

### 2. Set Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
cd backend
cp .env.example .env  # if .env.example exists
```

Or create the `.env` file manually with:

```env
# Google Gemini API Key
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Optional: LangSmith API Key for tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

### 3. Test the System

```bash
# Run debug script to check for issues
make debug

# Run system tests
make test

# Try CLI example
make cli
```

### 4. Start Development Servers

```bash
# Start both frontend and backend
make dev
```

## Troubleshooting

### Common Issues

1. **500 Error**: Usually means missing dependencies or API key
   - Run `make debug` to identify the issue
   - Ensure GEMINI_API_KEY is set correctly
   - Check that all dependencies are installed

2. **Import Errors**: Missing Python packages
   - Run `pip install -r backend/requirements.txt`
   - Ensure you're using Python 3.11+

3. **Frontend Build Issues**: Missing Node.js dependencies
   - Run `cd frontend && npm install`
   - Ensure Node.js 18+ is installed

### Debug Commands

```bash
# Check system health
make debug

# Test specific components
cd backend
python debug_system.py

# Check API endpoints
curl http://localhost:2024/api/health
```

## API Endpoints

Once running, the system provides these endpoints:

- `POST /api/refine-requirements` - Main requirements refinement
- `GET /api/health` - System health check
- `GET /api/agents` - Agent information

## Frontend Access

The frontend will be available at:
- Development: `http://localhost:5173`
- Production: `http://localhost:8123/app/`
