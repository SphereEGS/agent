from app import SpherexAgent
from app.config import logger
import signal
import sys
import logging
import traceback

def signal_handler(sig, frame):
    """Global signal handler to ensure graceful shutdown"""
    logger.info("Ctrl+C detected in main process, shutting down...")
    # Give a chance for cleanup
    try:
        # Some systems may handle SIGINT differently, so ensure we exit
        sys.exit(0)
    except SystemExit:
        # This is expected
        pass

if __name__ == "__main__":
    try:
        # Register the signal handler
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create and start the agent
        agent = SpherexAgent()
        agent.start()
    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)