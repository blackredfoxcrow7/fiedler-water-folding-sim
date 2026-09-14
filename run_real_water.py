import subprocess
import time
import sys
import os

def main():
    print("==================================================================")
    print("  Water Solvent & Peptide Model Simulator [REAL H2O WATER MODEL]")
    print("==================================================================")
    print("Starting high-fidelity H2O backend and frontend dev servers...")
    
    # 1. Launch the H2O Flask Backend on port 5002
    backend_process = subprocess.Popen(
        [sys.executable, "server_real_water.py"]
    )
    
    # Give the backend a moment to bind to port 5002
    time.sleep(1.0)
    print("✓ Real Water Backend Server started at:  http://localhost:5002")
    
    # 2. Launch the Vite React Frontend
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    
    frontend_process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd="client"
    )
    print("✓ Frontend Dev Server started. Checking port...")
    time.sleep(1.5)
    print("✓ Access the H2O real water simulator in your browser at:")
    print("  http://localhost:5173/?mode=realwater")
    print("Press Ctrl+C to terminate both servers.")
    print("==================================================================")
    
    try:
        while True:
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
