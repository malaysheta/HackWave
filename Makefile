.PHONY: help dev-frontend dev-backend dev test cli install debug

help:
	@echo "Multi-Agent Product Requirements Refinement System"
	@echo "=================================================="
	@echo "Available commands:"
	@echo "  make install        - Install all dependencies (frontend + backend)"
	@echo "  make dev-frontend   - Starts the frontend development server (Vite)"
	@echo "  make dev-backend    - Starts the backend development server (LangGraph)"
	@echo "  make dev            - Starts both frontend and backend development servers"
	@echo "  make test           - Run system tests"
	@echo "  make cli            - Run CLI example with sample query"
	@echo "  make debug          - Run debug script to identify issues"
	@echo "  make build          - Build frontend for production"

install:
	@echo "Installing dependencies..."
	@echo "Installing backend dependencies..."
	@cd backend && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	@cd frontend && npm install
	@echo "✅ All dependencies installed!"

dev-frontend:
	@echo "Starting frontend development server..."
	@cd frontend && npm run dev

dev-backend:
	@echo "Starting backend development server..."
	@cd backend && langgraph dev --allow-blocking

dev-backend-alt:
	@echo "Starting backend development server (alternative method)..."
	@cd backend && python start_backend.py

# Run frontend and backend concurrently
dev:
	@echo "Starting Multi-Agent Product Requirements Refinement System..."
	@echo "Frontend will be available at: http://localhost:5173"
	@echo "Backend will be available at: http://localhost:2024"
	@echo "API endpoints:"
	@echo "  - POST /api/refine-requirements"
	@echo "  - GET /api/health"
	@echo "  - GET /api/agents"
	@echo ""
	@make dev-frontend & make dev-backend

test:
	@echo "Running Multi-Agent System Tests..."
	@cd backend && python test_system.py

debug:
	@echo "Running Debug Script..."
	@cd backend && python debug_system.py

cli:
	@echo "Running CLI Example..."
	@cd backend && python examples/cli_research.py "What are the requirements for a mobile banking application?"

build:
	@echo "Building frontend for production..."
	@cd frontend && npm run build
	@echo "✅ Frontend built successfully!" 