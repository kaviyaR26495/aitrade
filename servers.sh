#!/bin/bash
# Helper script to manage backend and frontend servers
# Usage: ./servers.sh [start|stop|restart|status] [backend|frontend|celery|all]

VENV=".venv/bin/activate"
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
BACKEND_PID_FILE="/tmp/aitrade_backend.pid"
FRONTEND_PID_FILE="/tmp/aitrade_frontend.pid"
CELERY_WORKER_PID_FILE="/tmp/aitrade_celery_worker.pid"
CELERY_BEAT_PID_FILE="/tmp/aitrade_celery_beat.pid"

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

start_celery() {
  if [ -f "$CELERY_WORKER_PID_FILE" ] && kill -0 "$(cat $CELERY_WORKER_PID_FILE)" 2>/dev/null; then
    echo "Celery worker is already running (PID $(cat $CELERY_WORKER_PID_FILE))"
  else
    echo "Starting Celery worker..."
    source "$VENV"
    cd "$BACKEND_DIR"
    celery -A app.workers.celery_app worker \
      --loglevel=info \
      -Q data,ml \
      --concurrency=2 \
      --logfile=/tmp/aitrade_celery_worker.log &
    echo $! > "$CELERY_WORKER_PID_FILE"
    cd -
    echo "Celery worker started (PID $(cat $CELERY_WORKER_PID_FILE)) — log: /tmp/aitrade_celery_worker.log"
  fi

  if [ -f "$CELERY_BEAT_PID_FILE" ] && kill -0 "$(cat $CELERY_BEAT_PID_FILE)" 2>/dev/null; then
    echo "Celery Beat is already running (PID $(cat $CELERY_BEAT_PID_FILE))"
  else
    echo "Starting Celery Beat scheduler..."
    source "$VENV"
    cd "$BACKEND_DIR"
    celery -A app.workers.celery_app beat \
      --loglevel=info \
      --logfile=/tmp/aitrade_celery_beat.log &
    echo $! > "$CELERY_BEAT_PID_FILE"
    cd -
    echo "Celery Beat started (PID $(cat $CELERY_BEAT_PID_FILE)) — log: /tmp/aitrade_celery_beat.log"
  fi
}

stop_celery() {
  if [ -f "$CELERY_WORKER_PID_FILE" ]; then
    PID=$(cat "$CELERY_WORKER_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "Stopping Celery worker (PID $PID)..."
      kill "$PID"
      rm -f "$CELERY_WORKER_PID_FILE"
      echo "Celery worker stopped."
    else
      echo "Celery worker is not running."
      rm -f "$CELERY_WORKER_PID_FILE"
    fi
  else
    echo "Celery worker is not running."
  fi

  if [ -f "$CELERY_BEAT_PID_FILE" ]; then
    PID=$(cat "$CELERY_BEAT_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "Stopping Celery Beat (PID $PID)..."
      kill "$PID"
      rm -f "$CELERY_BEAT_PID_FILE"
      echo "Celery Beat stopped."
    else
      echo "Celery Beat is not running."
      rm -f "$CELERY_BEAT_PID_FILE"
    fi
  else
    echo "Celery Beat is not running."
  fi
}

status() {
  echo "=== Server Status ==="
  if [ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat $BACKEND_PID_FILE)" 2>/dev/null; then
    echo "Backend:       RUNNING (PID $(cat $BACKEND_PID_FILE)) - http://127.0.0.1:8000"
  else
    echo "Backend:       STOPPED"
  fi

  if [ -f "$FRONTEND_PID_FILE" ] && kill -0 "$(cat $FRONTEND_PID_FILE)" 2>/dev/null; then
    echo "Frontend:      RUNNING (PID $(cat $FRONTEND_PID_FILE)) - http://localhost:5173"
  else
    echo "Frontend:      STOPPED"
  fi

  if [ -f "$CELERY_WORKER_PID_FILE" ] && kill -0 "$(cat $CELERY_WORKER_PID_FILE)" 2>/dev/null; then
    echo "Celery Worker: RUNNING (PID $(cat $CELERY_WORKER_PID_FILE))"
  else
    echo "Celery Worker: STOPPED"
  fi

  if [ -f "$CELERY_BEAT_PID_FILE" ] && kill -0 "$(cat $CELERY_BEAT_PID_FILE)" 2>/dev/null; then
    echo "Celery Beat:   RUNNING (PID $(cat $CELERY_BEAT_PID_FILE))"
  else
    echo "Celery Beat:   STOPPED"
  fi
}

ACTION=${1:-start}
TARGET=${2:-all}

case "$ACTION" in
  start)
    [[ "$TARGET" == "backend" || "$TARGET" == "all" ]] && start_backend
    [[ "$TARGET" == "frontend" || "$TARGET" == "all" ]] && start_frontend
    [[ "$TARGET" == "celery" || "$TARGET" == "all" ]] && start_celery
    ;;
  stop)
    [[ "$TARGET" == "backend" || "$TARGET" == "all" ]] && stop_backend
    [[ "$TARGET" == "frontend" || "$TARGET" == "all" ]] && stop_frontend
    [[ "$TARGET" == "celery" || "$TARGET" == "all" ]] && stop_celery
    ;;
  restart)
    [[ "$TARGET" == "backend" || "$TARGET" == "all" ]] && stop_backend && start_backend
    [[ "$TARGET" == "frontend" || "$TARGET" == "all" ]] && stop_frontend && start_frontend
    [[ "$TARGET" == "celery" || "$TARGET" == "all" ]] && stop_celery && start_celery
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|status] [backend|frontend|celery|all]"
    exit 1
    ;;
esac
