#!/usr/bin/env python3
"""
Simple Personal AI Chatbot CLI
A file-first chatbot using Ollama with web search capabilities, thinking models, and tool use.

This is the main entry point for the refactored modular chatbot.
"""

from chat import ChatBot
from cli import ChatCLI
from pathlib import Path

def _load_dotenv():
    try:
        from dotenv import load_dotenv
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
    except Exception:
        # Dotenv is optional; ignore if not installed
        pass


def main():
    """Main entry point for the chatbot application."""
    _load_dotenv()
    chatbot = ChatBot()
    cli = ChatCLI(chatbot)
    cli.run()


if __name__ == "__main__":
    main()
