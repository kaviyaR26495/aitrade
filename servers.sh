#!/bin/bash
# Helper script to manage backend and frontend servers
# Usage: ./servers.sh [start|stop|restart|status] [backend|frontend|all]

VENV=".venv/bin/activate"
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
BACKEND_PID_FILE="/tmp/aitrade_backend.pid"
FRONTEND_PID_FILE="/tmp/aitrade_frontend.pid"

start_backend() {
  if [ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat $BACKEND_PID_FILE)" 2>/dev/null; then
    echo "Backend is already running (PID $(cat $BACKEND_PID_FILE))"
    return
  fi
  echo "Starting backend..."
  source "$VENV"
  cd "$BACKEND_DIR"
  uvicorn app.main:app --reload --port 8000 &
  echo $! > "$BACKEND_PID_FILE"
  cd -
  echo "Backend started (PID $(cat $BACKEND_PID_FILE)) at http://127.0.0.1:8000"
}

start_frontend() {
  if [ -f "$FRONTEND_PID_FILE" ] && kill -0 "$(cat $FRONTEND_PID_FILE)" 2>/dev/null; then
    echo "Frontend is already running (PID $(cat $FRONTEND_PID_FILE))"
    return
  fi
  echo "Starting frontend..."
  cd "$FRONTEND_DIR"
  npm run dev &
  echo $! > "$FRONTEND_PID_FILE"
  cd -
  echo "Frontend started (PID $(cat $FRONTEND_PID_FILE)) at http://localhost:5173"
}

stop_backend() {
  if [ -f "$BACKEND_PID_FILE" ]; then
    PID=$(cat "$BACKEND_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "Stopping backend (PID $PID)..."
      kill "$PID"
      rm -f "$BACKEND_PID_FILE"
      echo "Backend stopped."
    else
      echo "Backend is not running."
      rm -f "$BACKEND_PID_FILE"
    fi
  else
    echo "Backend is not running."
  fi
}

stop_frontend() {
  if [ -f "$FRONTEND_PID_FILE" ]; then
    PID=$(cat "$FRONTEND_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "Stopping frontend (PID $PID)..."
      kill "$PID"
      rm -f "$FRONTEND_PID_FILE"
      echo "Frontend stopped."
    else
      echo "Frontend is not running."
      rm -f "$FRONTEND_PID_FILE"
    fi
  else
    echo "Frontend is not running."
  fi
}

status() {
  echo "=== Server Status ==="
  if [ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat $BACKEND_PID_FILE)" 2>/dev/null; then
    echo "Backend:  RUNNING (PID $(cat $BACKEND_PID_FILE)) - http://127.0.0.1:8000"
  else
    echo "Backend:  STOPPED"
  fi

  if [ -f "$FRONTEND_PID_FILE" ] && kill -0 "$(cat $FRONTEND_PID_FILE)" 2>/dev/null; then
    echo "Frontend: RUNNING (PID $(cat $FRONTEND_PID_FILE)) - http://localhost:5173"
  else
    echo "Frontend: STOPPED"
  fi
}

ACTION=${1:-start}
TARGET=${2:-all}

case "$ACTION" in
  start)
    [[ "$TARGET" == "backend" || "$TARGET" == "all" ]] && start_backend
    [[ "$TARGET" == "frontend" || "$TARGET" == "all" ]] && start_frontend
    ;;
  stop)
    [[ "$TARGET" == "backend" || "$TARGET" == "all" ]] && stop_backend
    [[ "$TARGET" == "frontend" || "$TARGET" == "all" ]] && stop_frontend
    ;;
  restart)
    [[ "$TARGET" == "backend" || "$TARGET" == "all" ]] && stop_backend && start_backend
    [[ "$TARGET" == "frontend" || "$TARGET" == "all" ]] && stop_frontend && start_frontend
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|status] [backend|frontend|all]"
    exit 1
    ;;
esac
