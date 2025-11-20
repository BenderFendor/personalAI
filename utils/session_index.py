"""Maintain a manifest of chat sessions for sidebar navigation."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class SessionEntry:
    id: str
    file: str
    started_at: str
    title: str
    tokens_in: int = 0
    tokens_out: int = 0


class SessionIndex:
    """Simple JSON-backed index of chat sessions."""

    def __init__(self, index_path: str | Path = "chat_logs/index.json") -> None:
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {"version": 1, "sessions": []}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupted index; start fresh but keep a backup
            try:
                backup = self.index_path.with_suffix(".bak.json")
                self.index_path.rename(backup)
            except Exception:
                pass
            return {"version": 1, "sessions": []}

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    def add_session(
        self,
        session_id: str,
        file_path: str,
        title: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        started_at: datetime | None = None,
    ) -> None:
        data = self._load()
        entry = SessionEntry(
            id=session_id,
            file=file_path,
            started_at=(started_at or datetime.now()).isoformat(timespec="seconds"),
            title=title,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        # Prepend newest session
        sessions: List[Dict[str, Any]] = data.get("sessions", [])
        sessions.insert(0, asdict(entry))
        data["sessions"] = sessions[:500]  # cap to avoid unbounded growth
        self._atomic_write(data)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all indexed sessions."""
        data = self._load()
        return data.get("sessions", [])

    def get_session(self, session_id: str) -> Dict[str, Any] | None:
        """Get a specific session entry."""
        data = self._load()
        for session in data.get("sessions", []):
            if session["id"] == session_id:
                return session
        return None

    def rebuild_from_directory(self, chat_logs_dir: str | Path = "chat_logs") -> int:
        """Scan chat_logs directory and rebuild index from existing markdown files."""
        import re
        
        chat_logs_path = Path(chat_logs_dir)
        if not chat_logs_path.exists():
            return 0
        
        # Find all markdown files matching pattern chat_YYYYMMDD_HHMMSS.md
        md_files = sorted(chat_logs_path.glob("chat_*.md"), reverse=True)
        
        sessions = []
        for md_file in md_files:
            try:
                # Extract session_id from filename (e.g., chat_20251120_080926.md -> 20251120_080926)
                match = re.match(r'chat_(\d{8}_\d{6})\.md', md_file.name)
                if not match:
                    continue
                
                session_id = match.group(1)
                
                # Parse the markdown file to extract title and timestamp
                content = md_file.read_text(encoding='utf-8')
                
                # Extract started_at from first timestamp in file (format: ## ROLE [YYYY-MM-DD HH:MM:SS])
                timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', content)
                started_at = timestamp_match.group(1) if timestamp_match else session_id[:8] + 'T' + session_id[9:11] + ':' + session_id[11:13] + ':' + session_id[13:15]
                
                # Extract title from first USER message (up to 50 chars)
                title_match = re.search(r'## USER \[.*?\]\n\n(.*?)(?:\n\n|$)', content, re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()[:50]
                    # Remove markdown formatting
                    title = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)  # Remove links
                    title = re.sub(r'[*_`#]', '', title)  # Remove formatting chars
                else:
                    title = f"Chat {session_id}"
                
                entry = {
                    "id": session_id,
                    "file": str(md_file),
                    "started_at": started_at,
                    "title": title,
                    "tokens_in": 0,
                    "tokens_out": 0
                }
                sessions.append(entry)
                
            except Exception as e:
                print(f"Warning: Failed to parse {md_file.name}: {e}")
                continue
        
        # Write rebuilt index
        data = {"version": 1, "sessions": sessions[:500]}  # cap to 500 most recent
        self._atomic_write(data)
        
        return len(sessions)
