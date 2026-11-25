@echo off
REM Double-click to start the YOLO-World app. It opens in your browser at http://localhost:8501
REM The first start installs the libraries into a ".venv" folder here (about 10 minutes, needs internet).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First start: creating the Python environment and installing libraries...
    python -m venv .venv || (echo Python was not found. Install Python 3.10+ from python.org & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

if not exist "weights\yolo_world_s_custom.pt" (
    echo NOTE: weights\yolo_world_s_custom.pt not found - copy the trained models into the weights folder.
)

echo Starting the app... it will open in your browser at http://localhost:8501
echo Keep this window open while you use the app. Close it to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.address localhost
pause
