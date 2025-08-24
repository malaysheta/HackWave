# Docker Setup Guide for HackWave

## Quick Start

### 1. Set Up Environment Variables

**Option A: Use the setup script (Windows)**
```bash
setup_env.bat
```

**Option B: Create .env file manually**
Create a `.env` file in the project root with:
```bash
# API Keys - Replace with your actual keys
GEMINI_API_KEY=your_actual_gemini_api_key_here
LANGSMITH_API_KEY=your_actual_langsmith_api_key_here

# Optional: Enable blocking operations for development
BG_JOB_ISOLATED_LOOPS=true
```

### 2. Get API Keys

#### Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key
5. Paste it in your `.env` file

#### LangSmith API Key (Optional)
1. Go to [LangSmith](https://smith.langchain.com/)
2. Create an account
3. Get your API key from the dashboard
4. Add it to your `.env` file (or leave blank for development)

### 3. Build and Run

```bash
# Build the containers
docker compose build

# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs langgraph-api
```

### 4. Access the Application

- **Frontend**: http://localhost:8123
- **API Documentation**: http://localhost:8123/docs
- **PostgreSQL**: localhost:5433 (user: postgres, password: postgres)

## Troubleshooting

### Common Issues

#### 1. "GEMINI_API_KEY is not set"
- Make sure your `.env` file exists and contains the correct API key
- Restart the containers after creating the `.env` file

#### 2. "License verification failed"
- This is normal for development. The application will still work locally
- For production, you'll need a LangGraph license

#### 3. "BlockingError" in LangGraph Studio
- The `BG_JOB_ISOLATED_LOOPS=true` environment variable should resolve this
- If issues persist, check for any synchronous I/O operations in your code

#### 4. Container not starting
```bash
# Check logs
docker compose logs langgraph-api

# Restart containers
docker compose down
docker compose up -d
```

### Development Commands

```bash
# Rebuild after code changes
docker compose build --no-cache

# View real-time logs
docker compose logs -f langgraph-api

# Access container shell
docker compose exec langgraph-api bash

# Stop all services
docker compose down

# Remove all data (including database)
docker compose down -v
```

## Architecture

The application consists of three main services:

1. **langgraph-api**: Main application server (FastAPI + LangGraph)
2. **langgraph-postgres**: PostgreSQL database for persistence
3. **langgraph-redis**: Redis cache for session management

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `LANGSMITH_API_KEY` | LangSmith API key | No |
| `BG_JOB_ISOLATED_LOOPS` | Enable blocking operations | No |
| `REDIS_URI` | Redis connection string | Auto-set |
| `POSTGRES_URI` | PostgreSQL connection string | Auto-set |

## Ports

| Service | Internal Port | External Port |
|---------|---------------|---------------|
| API Server | 8000 | 8123 |
| PostgreSQL | 5432 | 5433 |
| Redis | 6379 | - |

## Support

If you encounter issues:
1. Check the logs: `docker compose logs`
2. Ensure all environment variables are set
3. Verify your API keys are valid
4. Check that ports 8123 and 5433 are available
