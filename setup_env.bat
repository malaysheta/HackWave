@echo off
echo Setting up environment for HackWave...
echo.

echo Please enter your Gemini API Key:
set /p GEMINI_API_KEY="GEMINI_API_KEY: "

echo Please enter your LangSmith API Key (optional, press Enter to skip):
set /p LANGSMITH_API_KEY="LANGSMITH_API_KEY: "

echo.
echo Creating .env file...

(
echo # API Keys
echo GEMINI_API_KEY=%GEMINI_API_KEY%
echo LANGSMITH_API_KEY=%LANGSMITH_API_KEY%
echo.
echo # Optional: Enable blocking operations for development
echo BG_JOB_ISOLATED_LOOPS=true
) > .env

echo.
echo .env file created successfully!
echo.
echo Now you can run: docker compose up -d
echo.
pause
