#!/usr/bin/env python3
"""
Test script to verify the refactored chatbot works correctly.
Run this to ensure all modules are properly imported and initialized.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config_manager import ConfigManager
        print("✓ config_manager imported")
    except ImportError as e:
        print(f"✗ Failed to import config_manager: {e}")
        return False
    
    try:
        from models import Message, ContextUsage, SearchResult, ToolCall
        print("✓ models imported")
    except ImportError as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    try:
        from tools import get_tool_definitions, ToolExecutor
        print("✓ tools module imported")
    except ImportError as e:
        print(f"✗ Failed to import tools: {e}")
        return False
    
    try:
        from utils import ContextCalculator, ChatLogger, DisplayHelper
        print("✓ utils module imported")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    try:
        from chat import ChatBot
        print("✓ chatbot imported")
    except ImportError as e:
        print(f"✗ Failed to import chatbot: {e}")
        return False
    
    try:
        from cli import ChatCLI
        print("✓ cli imported")
    except ImportError as e:
        print(f"✗ Failed to import cli: {e}")
        return False
    
    return True


def test_initialization():
    """Test that core components can be initialized."""
    print("\nTesting initialization...")
    
    try:
        from config_manager import ConfigManager
        config = ConfigManager()
        print(f"✓ ConfigManager initialized (model: {config.model})")
    except Exception as e:
        print(f"✗ Failed to initialize ConfigManager: {e}")
        return False
    
    try:
        from chat import ChatBot
        chatbot = ChatBot()
        print(f"✓ ChatBot initialized (session: {chatbot.current_session})")
    except Exception as e:
        print(f"✗ Failed to initialize ChatBot: {e}")
        return False
    
    return True


def test_models():
    """Test that models work correctly."""
    print("\nTesting models...")
    
    try:
        from models import Message, ContextUsage
        
        # Test Message
        msg = Message(role='user', content='Hello')
        assert msg.role == 'user'
        assert msg.content == 'Hello'
        assert 'timestamp' in msg.to_dict()
        print("✓ Message model works")
        
        # Test ContextUsage
        usage = ContextUsage(current_tokens=100, max_tokens=1000)
        assert usage.percentage == 10.0
        assert usage.remaining_tokens == 900
        assert usage.color == 'green'
        print("✓ ContextUsage model works")
        
    except Exception as e:
        print(f"✗ Failed model tests: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Personal AI Chatbot - Refactored Module Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization),
        ("Model Test", test_models),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! The refactored chatbot is ready to use.")
        print("\nRun 'python main.py' to start the chatbot.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
