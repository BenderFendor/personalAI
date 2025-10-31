#!/usr/bin/env python3
"""
Simple Personal AI Chatbot CLI
A file-first chatbot using Ollama with web search capabilities, thinking models, and tool use.

This is the main entry point for the refactored modular chatbot.
"""

from chatbot import ChatBot
from cli import ChatCLI


def main():
    """Main entry point for the chatbot application."""
    chatbot = ChatBot()
    cli = ChatCLI(chatbot)
    cli.run()


if __name__ == "__main__":
    main()
