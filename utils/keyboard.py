"""Keyboard input handling with special key detection."""

import sys
import tty
import termios
from typing import Optional, Tuple


class KeyboardHandler:
    """Handle keyboard input with special key detection."""
    
    # Special key codes
    CTRL_TAB = '\t'  # Note: Ctrl+Tab may be captured by terminal
    CTRL_C = '\x03'
    CTRL_D = '\x04'
    CTRL_B = '\x02'  # Ctrl-b (safer toggle than Ctrl-])
    ESC = '\x1b'
    ENTER = '\r'
    UP = '\x1b[A'
    DOWN = '\x1b[B'
    LEFT = '\x1b[C'
    RIGHT = '\x1b[D'
    
    @staticmethod
    def get_key() -> Optional[str]:
        """Get a single keypress.
        
        Returns:
            Key code or None if interrupted
        """
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                
                # Check for escape sequences (arrow keys, etc.)
                if ch == '\x1b':
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        return f'\x1b[{ch3}'
                    return ch + ch2
                
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except:
            return None
    
    @staticmethod
    def is_ctrl_tab(key: str) -> bool:
        """Check if key is Ctrl+Tab combination.
        
        Note: Some terminals don't support Ctrl+Tab, so we also check for Ctrl+]
        as an alternative keybinding.
        
        Args:
            key: Key code
            
        Returns:
            True if Ctrl+Tab or Ctrl+]
        """
        return key in ['\t', '\x1d']  # Tab or Ctrl+]

    @staticmethod
    def is_ctrl_toggle(key: str) -> bool:
        """Check if key is the sidebar toggle (Ctrl-b).

        Args:
            key: Key code

        Returns:
            True if Ctrl-b detected.
        """
        return key == KeyboardHandler.CTRL_B
