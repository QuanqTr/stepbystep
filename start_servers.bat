@echo off
start cmd /k "uvicorn api.index:app --reload --port 8000"
start cmd /k "npm run dev"
echo Servers started. Backend at port 8000, Frontend at port 3000.
pause
