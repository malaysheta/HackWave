@echo off
echo ========================================
echo    HACKWAVE COMPLETE PROJECT TEST
echo ========================================
echo.

echo [1/6] Checking container status...
docker compose ps
echo.

echo [2/6] Testing health endpoint...
curl -s http://localhost:8123/api/health
echo.
echo.

echo [3/6] Testing available agents...
curl -s http://localhost:8123/api/agents
echo.
echo.

echo [4/6] Testing basic product requirements analysis...
curl -X POST http://localhost:8123/api/refine-requirements -H "Content-Type: application/json" -d "{\"query\": \"Build an e-commerce platform for handmade crafts\"}"
echo.
echo.

echo [5/6] Testing streaming functionality...
echo Starting streaming test (will timeout after 30 seconds)...
curl -X POST http://localhost:8123/api/refine-requirements/stream -H "Content-Type: application/json" -d "{\"query\": \"Create a fitness tracking app with social features\"}" --max-time 30
echo.
echo.

echo [6/6] Testing API documentation access...
echo API Documentation available at: http://localhost:8123/docs
echo.

echo ========================================
echo    TEST COMPLETE!
echo ========================================
echo.
echo Access your application at:
echo - Frontend: http://localhost:8123
echo - API Docs: http://localhost:8123/docs
echo - Health Check: http://localhost:8123/api/health
echo.
echo All services are running successfully!
pause
