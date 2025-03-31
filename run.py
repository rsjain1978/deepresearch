import uvicorn
from termcolor import colored
import webbrowser
import os
from dotenv import load_dotenv
import time

def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for required environment variables
        required_vars = ['OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(colored(f"Error: Missing required environment variables: {', '.join(missing_vars)}", "red"))
            return

        # Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("data/chroma", exist_ok=True)
        os.makedirs("static", exist_ok=True)

        print(colored("Starting Deep Research Application...", "green"))
        print(colored("✓ Environment variables loaded", "green"))
        print(colored("✓ Directories initialized", "green"))
        
        # Open browser after a short delay to ensure server is up
        def open_browser():
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open("http://localhost:8000")
            print(colored("✓ Browser opened at http://localhost:8000", "green"))

        # Start the server with hot reload
        config = uvicorn.Config(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["app"],
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        
        # Open browser in a separate thread
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Start the server
        server.run()

    except Exception as e:
        print(colored(f"Error starting application: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    main() 