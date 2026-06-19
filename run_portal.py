import http.server
import socketserver
import os
import subprocess
import json
import sys

# Configuration
PORT = 8000
WEB_DIR = os.path.join(os.path.dirname(__file__), "web_portal")
MAIN_APP = os.path.join(os.path.dirname(__file__), "final_pred.py")

class PortalHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_POST(self):
        if self.path == '/launch':
            try:
                # Launch the main application separately
                # We use subprocess.Popen to run it as a detached process
                print(f"Attempting to launch: {MAIN_APP}")
                
                # Determine python executable
                python_exe = sys.executable
                
                # On Windows, we can use start to run it in a new window
                if os.name == 'nt':
                     subprocess.Popen([python_exe, MAIN_APP], 
                                     creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen([python_exe, MAIN_APP])

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"status": "success", "message": "Application launched"}
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                print(f"Error launching app: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(response).encode())

def run_server():
    # Make sure we are in the right directory to serve files
    os.chdir(os.path.dirname(__file__))
    
    with socketserver.TCPServer(("", PORT), PortalHandler) as httpd:
        print(f"--- EchoSign Web Portal ---")
        print(f"Server started at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop the portal server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down portal...")
            httpd.shutdown()

if __name__ == "__main__":
    # Check if final_pred.py exists
    if not os.path.exists(MAIN_APP):
        print(f"Error: Could not find main application at {MAIN_APP}")
        sys.exit(1)
        
    run_server()
