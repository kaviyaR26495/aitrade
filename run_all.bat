@echo off
echo Starting databases via docker-compose...
docker compose up -d

echo Starting Backend in a new window...
start cmd /k "cd backend && ..\.venv\Scripts\activate.bat && uvicorn app.main:app --reload --port 8000"

echo Starting Frontend in a new window...
start cmd /k "set PATH=C:\Program Files\nodejs;%PATH% && cd frontend && npm run dev"

echo Project is starting!
echo Backend API: http://127.0.0.1:8000
echo Frontend: http://localhost:5173
