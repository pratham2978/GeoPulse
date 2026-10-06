"""
GeoPulse AI Platform - Full Website Launcher
Starts both the FastAPI Backend (Port 8000) and the Vite Frontend (Port 5173) together.
"""

import subprocess
import sys
import os
import time

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 65)
    print("  [+] GEOPULSE AI - GLOBAL INTELLIGENCE PLATFORM")
    print("=" * 65)
    print("Starting Backend & Frontend services...\n")

    # 1. Start Backend
    print("[1/2] Launching Python FastAPI Backend on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "backend/app.py"],
        cwd=root_dir
    )

    time.sleep(2)

    # 2. Start Frontend
    print("[2/2] Launching Vite React Frontend on http://localhost:5173 ...")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=root_dir
    )

    print("\n" + "=" * 65)
    print("  [SUCCESS] FULL WEBSITE IS RUNNING!")
    print("=" * 65)
    print("  -> Frontend Dashboard: http://localhost:5173")
    print("  -> Backend REST API:   http://localhost:8000")
    print("  -> API Swagger Docs:   http://localhost:8000/docs")
    print("=" * 65)
    print("Press Ctrl+C in this terminal to stop both servers.\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Servers stopped cleanly.")

if __name__ == "__main__":
    main()
