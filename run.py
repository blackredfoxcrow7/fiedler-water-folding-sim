import subprocess
import time
import sys
import os

def main():
    print("==================================================================")
    print("       Water Solvent & Peptide Model Agent Simulator")
    print("==================================================================")
    print("Starting backend and frontend development servers...")
    
    # 1. Launch the Flask Backend on port 5000
    backend_process = subprocess.Popen(
        [sys.executable, "server.py"]
    )
    
    # Give the backend a moment to bind to port 5000
    time.sleep(1.0)
    print("✓ Backend Server started at:  http://localhost:5000")
    
    # 2. Launch the Vite React Frontend
    # Determine command based on platform (npx.cmd on windows, npm on unix)
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    
    frontend_process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd="client"
    )
    print("✓ Frontend Dev Server started. Checking port...")
    time.sleep(1.5)
    print("✓ Access the simulator in your browser at: http://localhost:5173")
    print("Press Ctrl+C to terminate both servers.")
    print("==================================================================")
    
    try:
        # Keep monitoring output logs
        while True:
            # We can print logs if needed, or just sleep
            # Let's check if either process terminated
            if backend_process.poll() is not None:
                print("Backend server terminated unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("Frontend server terminated unexpectedly.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping processes...")
    finally:
        # Graceful cleanup
        backend_process.terminate()
        frontend_process.terminate()
        try:
            backend_process.wait(timeout=2)
            frontend_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            backend_process.kill()
            frontend_process.kill()
        print("Both servers have been stopped. Goodbye!")

if __name__ == '__main__':
    main()
