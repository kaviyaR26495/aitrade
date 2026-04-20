# Start all (Redis + Backend + Frontend + Celery Worker + Celery Beat)

sudo systemctl start redis-server
./servers.sh start

# Stop all

./servers.sh stop

# Restart specific server

./servers.sh restart backend
./servers.sh restart frontend
./servers.sh restart celery

# Check status

./servers.sh status

# Targets: backend | frontend | celery | all (default)

# Note: Redis must be running before starting Celery. If data sync appears stuck, check Redis:

# sudo systemctl status redis-server

# sudo systemctl start redis-server

# ./servers.sh restart celery
