"""Chat logging utilities."""

from pathlib import Path
from typing import List
from models import Message


class ChatLogger:
    """Handles saving chat sessions to markdown files."""
    
    def __init__(self, logs_dir: str = "chat_logs"):
        """Initialize chat logger.
        
        Args:
            logs_dir: Directory for chat logs
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
    
    def save_session(
        self,
        session_id: str,
        messages: List[Message],
        config: dict
    ) -> Path:
        """Save chat session to markdown file.
        
        Args:
            session_id: Unique session identifier
            messages: List of messages in the session
            config: Configuration dictionary
            
        Returns:
            Path to saved log file
        """
        log_file = self.logs_dir / f"chat_{session_id}.md"
        
        with open(log_file, 'w') as f:
            # Write header
            f.write(f"# Chat Log - {session_id}\n\n")
            f.write(f"**Model:** {config.get('model', 'Unknown')}\n")
            f.write(f"**Temperature:** {config.get('temperature', 'Unknown')}\n")
            f.write(f"**Tools Enabled:** {config.get('tools_enabled', False)}\n")
            f.write(f"**Thinking Enabled:** {config.get('thinking_enabled', False)}\n\n")
            f.write("---\n\n")
            
            # Write messages
            for msg in messages:
                role = msg.role.upper()
                content = msg.content
                timestamp = msg.timestamp
                
                f.write(f"## {role} [{timestamp}]\n\n")
                
                # Include thinking if available
                if msg.thinking:
                    f.write("### Thinking Process\n\n")
                    f.write(f"```\n{msg.thinking}\n```\n\n")
                
                f.write(f"{content}\n\n")
                
                # Include sources if available
                if msg.sources:
                    f.write("**Sources:**\n")
                    for i, source in enumerate(msg.sources, 1):
                        f.write(f"{i}. [{source['title']}]({source['url']})\n")
                        f.write(f"   - {source['snippet']}\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        return log_file
