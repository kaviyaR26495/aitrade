param(
    [string]$Action = "start"
)

if ($Action -eq "start") {
    Write-Host "Starting databases via docker compose..."
    docker compose up -d

    Write-Host "Starting Backend in a new window..."
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -NoExit -Command `"cd backend; & '..\.venv\Scripts\python.exe' -m uvicorn app.main:app --reload --port 8000`""
    
    Write-Host "Starting Frontend in a new window..."
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -NoExit -Command `"cd frontend; & 'C:\Program Files\nodejs\npm.cmd' run dev`""
    
    Write-Host "Project is starting! Backend API: http://127.0.0.1:8000 | Frontend: http://localhost:5173"
} elseif ($Action -eq "stop") {
    Write-Host "Stopping Docker containers..."
    docker compose stop
    Write-Host "To stop the backend and frontend, please close their respective powershell windows manually."
} else {
    Write-Host "Usage: .\run_all.ps1 [start|stop]"
}
