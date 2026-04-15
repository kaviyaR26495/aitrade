# Start servers

To start the project on Windows using Command Prompt, you can run the batch script:
```cmd
.\run_all.bat
```
This will launch Docker resources in the background, and open two new CMD windows for the backend and frontend servers.

- **Backend API**: `http://127.0.0.1:8000`
- **Frontend App**: `http://localhost:5173`

# Stop servers

If you wish to stop Docker containers, you can manually run:
```cmd
docker compose stop
```
If you wish to stop the backend or frontend servers, simply close their respective spawned CMD windows.

---
*(Note: A Linux/macOS equivalent script is available in `./servers.sh` if needed).*
