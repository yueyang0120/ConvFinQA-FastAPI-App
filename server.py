#!/usr/bin/env python
"""
Run the Financial QA API server.
"""

import os
import sys
import argparse
import uvicorn

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial QA API Server")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the server script."""
    args = parse_args()
    
    # Start the server
    print(f"Starting Financial QA API server on {args.host}:{args.port}...")
    uvicorn.run("src.main:app", host=args.host, port=args.port, log_level=args.log_level, reload=True)


if __name__ == "__main__":
    main() 